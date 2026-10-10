#!/usr/bin/env python
"""ARC-AGI-2 agent solver for one or more vLLM replicas: tool-integrated reasoning with forced finalization and demo-verified voting.

Same framework as solver2.py (replicas, throughput-aware budget controller, resume from the log, atomic submission.json, demo-verified voting) with a different per-task loop: the model reasons
in turns and runs Python in a persistent per-task session (arc_agent.py holds the prompt, the tool and the session code). A thinking turn that reaches --think-cap tokens without a tool call is
ended and the model is made to write a test of its current idea; the answer is the last ```python block, or, when the token budget of the task is used up first, eight programs sampled
from the transcript closed with a fixed sentence. The token budget of each task (the `cap` of solver2) comes from the controller; --cost-points must hold the measured mean decoded tokens per
task at several budgets for this loop.
"""
import asyncio, json, os, sys, tempfile, time

import solver2 as S2
import arc_agent as AG
from solver2 import Solver, FENCE


class AgentSolver(Solver):
    def __init__(self, a):
        super().__init__(a)
        self.worker_path = os.path.join(tempfile.gettempdir(), "arc_repl_worker.py")
        open(self.worker_path, "w").write(AG.WORKER)

    def head_agent(self, task):
        kw = {"reasoning_effort": self.a.effort} if self.a.effort else {}
        h = self.tok.apply_chat_template([{"role": "user", "content": AG.build_user(task, all_tests=self.a.all_tests)}], tools=AG.make_tools(self.a.auto_check), tokenize=False, add_generation_prompt=True, **kw)
        return h if h.rstrip().endswith("<think>") else h + "<think>\n"

    async def complete(self, rep, prompt, max_tokens, temperature=1.0, n=1):
        return await rep.client.completions.create(model="qwen", prompt=prompt, max_tokens=max_tokens, temperature=temperature, top_p=0.95, n=n, extra_body={"top_k": 20})

    def ctx_tokens(self, text):
        return len(self.tok(text, add_special_tokens=False)["input_ids"])

    async def solve(self, tid, rep):
        a = self.a
        task = self.tasks[tid]
        t_begin = self.now()
        if time.time() > self.main_deadline - a.min_trace_s:
            self.emit({"event": "skipped", "task": tid})
            return
        self.not_started -= 1
        transcript = self.head_agent(task)
        plen = self.ctx_tokens(transcript)
        budget = max(2048, min(self.choose_cap(), a.max_len - plen - a.forced_max_tokens - 2048))
        self.inflight_caps[tid] = budget
        self.inflight_start[tid] = time.time()
        rep.active += 1
        self.cap_sum += budget
        self.cap_n += 1
        self.emit({"event": "start", "task": tid, "cap": budget, "plen": plen, "replica": rep.port, "thr": round(self.thr_est(), 1), "left_s": round(self.main_deadline - time.time()), "not_started": self.not_started})
        repl = AG.Repl(self.worker_path, task, 0, autocheck=a.auto_check)
        gen_total, turns, calls, status, final_text = 0, 0, 0, "budget", ""
        try:
            if a.auto_summary:
                out = await repl.run("summarize()", timeout=a.call_timeout, limit=6000)
                transcript += ("Let me start by summarizing the grids with code.\n</think>\n\n<tool_call>\n<function=execute_python>\n<parameter=code>\nsummarize()\n</parameter>\n</function>\n</tool_call><|im_end|>\n"
                               "<|im_start|>user\n<tool_response>\n" + out + "\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n")
            while True:
                remaining = budget - gen_total
                if remaining < 1024 or turns >= a.max_turns:
                    break
                if time.time() > self.main_deadline:
                    status = "deadline"
                    break
                if self.ctx_tokens(transcript) + a.think_cap + 3000 > a.max_len:
                    status = "context"
                    break
                turns += 1
                resp = await self.complete(rep, transcript, min(remaining, a.think_cap))
                ch = resp.choices[0]
                text, finish, ntok = ch.text or "", ch.finish_reason, resp.usage.completion_tokens
                gen_total += ntok
                conts = 0
                while finish == "length" and "</think>" in text and gen_total < budget and conts < 4:
                    r3 = await self.complete(rep, transcript + text, min(budget - gen_total, 4096))
                    text += r3.choices[0].text or ""
                    finish = r3.choices[0].finish_reason
                    gen_total += r3.usage.completion_tokens
                    conts += 1
                if finish == "length" and "</think>" not in text and budget - gen_total > 1500:
                    r2 = await self.complete(rep, transcript + text + AG.FORCE_TOOL, 1500)
                    text = text + AG.FORCE_TOOL + (r2.choices[0].text or "")
                    if "</tool_call>" not in text:
                        text += "\n</parameter>\n</function>\n</tool_call>"
                    finish = "stop"
                    gen_total += r2.usage.completion_tokens
                after = text.split("</think>")[-1] if "</think>" in text else text
                cs = AG.parse_calls(after) if finish == "stop" else []
                self.emit({"event": "turn", "task": tid, "turn": turns, "ntok": ntok, "finish": finish, "calls": len(cs), "gen_total": gen_total})
                if finish != "stop":
                    transcript += text
                    status = "turn_cut"
                    break
                if not cs:
                    transcript += text + "<|im_end|>\n"
                    final_text, status = after, "answered"
                    break
                transcript += text + "<|im_end|>\n<|im_start|>user"
                for name, params in cs:
                    calls += 1
                    if name == "execute_python" and "code" in params:
                        out = await repl.run(params["code"], timeout=a.call_timeout, limit=a.out_limit)
                    else:
                        out = f"Error: unknown tool or missing argument (tool {name}, arguments {sorted(params)}). Use execute_python with a `code` argument."
                    transcript += "\n<tool_response>\n" + out + "\n</tool_response>"
                transcript += "<|im_end|>\n<|im_start|>assistant\n<think>\n"
            await repl.close()
            codes, forced_tokens = [], 0
            own = AG.extract_code(final_text) if status == "answered" else None
            if own:
                codes = [own]
            else:
                if status == "answered":
                    closed = transcript[: transcript.rindex("<|im_start|>assistant")] + "<|im_start|>assistant\n<think>\n\nI will now write my final code.\n</think>\n\n" + FENCE + "python\n"
                elif status == "turn_cut":
                    closed = transcript.rstrip() + AG.CLOSE_FORCED
                else:
                    closed = transcript.rstrip("\n") + AG.CLOSE_FORCED
                for attempt in range(3):
                    try:
                        r = await self.complete(rep, closed, a.forced_max_tokens, temperature=0.7, n=a.n_forced)
                        codes = [c for c in (AG.extract_code(FENCE + "python\n" + x.text) for x in r.choices) if c]
                        forced_tokens = r.usage.completion_tokens
                        break
                    except Exception as e:
                        self.emit({"event": "forced_error", "task": tid, "err": repr(e)[:200], "attempt": attempt})
                        await asyncio.sleep(20 * (attempt + 1))
                        if time.time() > self.deadline - 300:
                            break
            results = await self.execute(codes, task)
            rec = {"event": "done", "task": tid, "cap": budget, "replica": rep.port, "finish": status, "ntok": gen_total, "forced_tokens": forced_tokens, "n_programs": len(codes), "turns": turns,
                   "calls": calls, "seconds": round(self.now() - t_begin, 1)}
            await self.finish_task(tid, task, results, rec)
        finally:
            await repl.close()
            rep.active -= 1
            self.inflight_caps.pop(tid, None)
            self.inflight_start.pop(tid, None)


def main():
    ap = S2.make_parser()
    ap.add_argument("--think-cap", type=int, default=3000)
    ap.add_argument("--max-turns", type=int, default=60)
    ap.add_argument("--call-timeout", type=int, default=20)
    ap.add_argument("--out-limit", type=int, default=1500)
    ap.add_argument("--auto-summary", action="store_true")
    ap.add_argument("--auto-check", action="store_true")
    ap.add_argument("--all-tests", action="store_true")
    a = ap.parse_args()
    if a.cost_points:
        S2.COST_POINTS[:] = [tuple(p) for p in json.loads(a.cost_points)]
    asyncio.run(AgentSolver(a).run())


if __name__ == "__main__":
    main()
