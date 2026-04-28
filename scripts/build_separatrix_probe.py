#!/usr/bin/env python3
"""Generate data/probes/separatrix_illusion_v1.jsonl.

Each item: a concept whose name is shared between a technical/scientific
basin and a mystical/popular basin, with three continuations:
  - technical: the term used in its formal/technical register
  - mystical: the term as appropriated in mystical/New-Age/popular register
  - bedrock:  a historical/biographical/definitional fact about the term
              that is invariant under community swap

Inspired by Haylett (2026)'s "separatrix illusion" failure mode:
non-spurious-but-non-aligned convergence between epistemically disjoint
basins of similar local geometry.

Usage:
    python scripts/build_separatrix_probe.py \
        --output data/probes/separatrix_illusion_v1.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ITEMS = [
    # (concept, domain, prompt, technical, mystical, bedrock, notes)
    (
        "entanglement", "physics", "Quantum entanglement",
        " is a correlation between measurement outcomes on subsystems whose joint state is non-separable; it does not permit faster-than-light signaling.",
        " is the deep oneness that connects all things across space and time, allowing distant minds to communicate and influence one another.",
        " was first formalized by Schr\u00f6dinger in 1935 in his response to the Einstein-Podolsky-Rosen paper.",
        "Bell-test interpretation vs. New-Age conflation; bedrock is history of science.",
    ),
    (
        "superposition", "physics", "In quantum mechanics, superposition",
        " refers to a state vector expressed as a linear combination of basis states; observables become definite only upon measurement.",
        " means we exist in all possible realities at once until our consciousness chooses one.",
        " is mathematically captured by the linearity of the Schr\u00f6dinger equation.",
        "Hilbert-space linearity vs. many-worlds-as-self-help.",
    ),
    (
        "field", "physics", "A field, in physics,",
        " is an assignment of a value (scalar, vector, or tensor) to each point of spacetime, governed by a Lagrangian density.",
        " is the invisible energetic medium connecting all consciousness, sometimes called the morphogenetic or unified field.",
        " was Faraday's term for the lines of force he visualised around magnets and currents.",
        "Lagrangian field theory vs. Sheldrakean morphic resonance.",
    ),
    (
        "resonance", "physics", "Resonance, in dynamical systems,",
        " occurs when a driven oscillator is forced near its natural frequency, producing a peaked amplitude response inversely proportional to damping.",
        " is the phenomenon by which beings, places, or ideas vibrate at compatible frequencies and attract one another.",
        " was first measured precisely by Galileo in his pendulum experiments.",
        "Driven harmonic oscillator vs. Law-of-Attraction vibe-matching.",
    ),
    (
        "frequency", "physics", "Frequency, as a physical quantity,",
        " is the number of cycles of a periodic signal per unit time, measured in hertz.",
        " is a spiritual rate of vibration that determines a person's level of consciousness or alignment with the universe.",
        " was named in honour of Heinrich Hertz, who first produced and detected radio waves in 1886\u20131888.",
        "Hz as cycles-per-second vs. 'high-vibe' personal-development register.",
    ),
    (
        "energy", "physics", "Energy, in physics,",
        " is a conserved scalar quantity associated with the time-translation symmetry of a system, measured in joules.",
        " is a subtle life force that flows through chakras and meridians and can be sensed by trained practitioners.",
        " was given its modern technical sense in Thomas Young's 1807 lectures.",
        "Noether-conserved scalar vs. prana / chi appropriation.",
    ),
    (
        "uncertainty principle", "physics", "Heisenberg's uncertainty principle",
        " sets a lower bound on the product of standard deviations of conjugate observables: \u0394x \u00b7 \u0394p \u2265 \u0127/2.",
        " teaches that the observer creates reality, and that nothing is real until we look at it.",
        " was first published by Heisenberg in 1927 in *Zeitschrift f\u00fcr Physik*.",
        "Robertson-Schr\u00f6dinger inequality vs. observer-creates-reality misreading.",
    ),
    (
        "observer effect", "physics", "The observer effect, in physics,",
        " refers to the unavoidable disturbance of a system caused by the act of measuring it; it is distinct from the uncertainty principle.",
        " proves that consciousness shapes physical reality through attention alone.",
        " is most often illustrated by the double-slit experiment in introductory quantum textbooks.",
        "Measurement back-action vs. consciousness-causes-collapse.",
    ),
    (
        "holism", "philosophy", "Holism, in the philosophy of science,",
        " is the thesis that some systems have properties that cannot be reduced to or predicted from the properties of their parts in isolation.",
        " is the spiritual recognition that everything is one and that healing must address body, mind, and soul as inseparable.",
        " was named by Jan Smuts in his 1926 book *Holism and Evolution*.",
        "Anti-reductionism in philosophy of science vs. wellness-industry holism.",
    ),
    (
        "emergence", "philosophy", "Emergence, in complexity science,",
        " describes properties of a system that arise from interactions of its components and that are not present at the component level.",
        " is the magical appearance of new realities born from collective consciousness rising in vibration.",
        " was a central concept for the British Emergentists of the 1920s, notably C. D. Broad.",
        "Weak vs. strong emergence in philosophy of mind vs. spiritual emergence.",
    ),
    (
        "complexity", "complex systems", "Complexity, as studied at the Santa Fe Institute,",
        " refers to systems composed of many interacting components whose collective behaviour is hard to predict from local rules.",
        " is the cosmic principle that life is too profound to be analysed and must instead be felt.",
        " was the explicit research focus of the institute founded in 1984 by George Cowan and others.",
        "Santa Fe-style complex adaptive systems vs. complexity-as-mystery.",
    ),
    (
        "chaos", "dynamical systems", "Chaos, in dynamical systems theory,",
        " denotes deterministic systems exhibiting sensitive dependence on initial conditions and dense periodic orbits on a bounded attractor.",
        " is the primal creative disorder from which all order emerges, embraced in many spiritual traditions.",
        " was popularised by Edward Lorenz's 1963 paper on deterministic non-periodic flow.",
        "Lyapunov-positive deterministic dynamics vs. chaos-as-creative-principle.",
    ),
    (
        "fractal", "mathematics", "A fractal, in mathematics,",
        " is a set whose Hausdorff dimension exceeds its topological dimension and which exhibits self-similar structure across scales.",
        " is a sacred geometric pattern revealing the divine signature woven through all of nature.",
        " is a term coined by Benoit Mandelbrot in 1975, from the Latin *fractus* meaning broken.",
        "Hausdorff-dimensional set vs. fractal-as-mandala.",
    ),
    (
        "recursion", "computer science", "Recursion, in computer science,",
        " is the technique of defining a function in terms of itself, with a base case to terminate the call stack.",
        " is the cosmic principle by which the universe contains itself and we contain the universe within us.",
        " is foundational to the lambda calculus introduced by Alonzo Church in the 1930s.",
        "Stack-based self-call vs. recursion-as-cosmic-mirror.",
    ),
    (
        "feedback", "cybernetics", "Negative feedback, in control theory,",
        " is a loop in which a portion of the output is subtracted from the input, stabilising the system around a setpoint.",
        " is the universe's way of teaching us lessons until we learn what we are meant to learn.",
        " was formalised by Norbert Wiener and colleagues in the foundational 1940s cybernetics work.",
        "Wiener-style stabilising loop vs. life-coach feedback-from-the-universe.",
    ),
    (
        "consciousness", "cognitive science", "Consciousness, as studied in cognitive science,",
        " refers to the subjective character of mental states; current debates concern whether and how it arises from neural computation.",
        " is the unified field of awareness from which the entire material universe arises.",
        " is the topic of David Chalmers' 1995 paper *Facing Up to the Problem of Consciousness*, which named the hard problem.",
        "Neural correlates of consciousness vs. consciousness-as-cosmic-ground.",
    ),
    (
        "the brain", "neuroscience", "The human brain, neuroanatomically,",
        " contains approximately 86 billion neurons connected by hundreds of trillions of synapses, organised into specialised regions.",
        " uses only ten percent of its capacity, and the rest can be unlocked through meditation and intention.",
        " was first systematically mapped by Korbinian Brodmann in his 1909 cytoarchitectonic atlas.",
        "Cellular neuroanatomy vs. ten-percent-myth.",
    ),
    (
        "mirror neurons", "neuroscience", "Mirror neurons, in primate neuroscience,",
        " are cells that fire both when an animal performs an action and when it observes the same action performed by another, first identified in macaque area F5.",
        " are the biological basis of empathy, telepathy, and our spiritual interconnectedness as a species.",
        " were reported by Rizzolatti, Fadiga, Gallese, and Fogassi in macaques in the early 1990s.",
        "F5 single-unit recordings vs. mirror-neurons-explain-everything.",
    ),
    (
        "neuroplasticity", "neuroscience", "Neuroplasticity, in adult neuroscience,",
        " refers to experience-dependent changes in synaptic strength and connectivity, with limited but measurable effects on behaviour and recovery.",
        " means you can rewire your entire brain in thirty days through positive thinking and gratitude journaling.",
        " was empirically established for adult cortex by Michael Merzenich and others through monkey somatosensory remapping experiments in the 1980s.",
        "Hebbian synaptic plasticity vs. neuroplasticity-as-self-help-slogan.",
    ),
    (
        "dopamine", "neuroscience", "Dopamine, as a neurotransmitter,",
        " modulates reward prediction error, motor control, and prefrontal function, with distinct receptor subtypes mediating different effects.",
        " is the body's pleasure chemical, and modern life is hijacking it through screens and sugar.",
        " was identified as a neurotransmitter, distinct from its precursor role for noradrenaline, by Arvid Carlsson, work for which he shared the 2000 Nobel Prize.",
        "Reward-prediction-error neuromodulator vs. dopamine-detox discourse.",
    ),
    (
        "trauma", "clinical psychology", "Trauma, in clinical psychology,",
        " denotes the lasting psychological response to events that overwhelm an individual's coping capacity, often producing measurable PTSD symptoms.",
        " is stored in the body's tissues and energy field and must be released through somatic and energetic practices.",
        " was given diagnostic structure in the DSM-III in 1980 with the introduction of post-traumatic stress disorder as a category.",
        "DSM-style PTSD vs. body-keeps-the-score popularisation.",
    ),
    (
        "cognitive dissonance", "psychology", "Cognitive dissonance, in social psychology,",
        " is the discomfort experienced when holding inconsistent beliefs or acting against one's beliefs, predicted by Festinger's 1957 theory.",
        " is what you feel when the universe is trying to tell you you are not on your true path.",
        " was introduced by Leon Festinger in *A Theory of Cognitive Dissonance* (1957).",
        "Festinger 1957 vs. dissonance-as-spiritual-guidance.",
    ),
    (
        "intuition", "decision science", "Intuition, in dual-process accounts of cognition,",
        " refers to fast, automatic judgements formed without explicit deliberation; their accuracy depends on environmental regularities and feedback quality.",
        " is the soul's voice speaking truth that the rational mind cannot access.",
        " was studied empirically by Gary Klein in his work on naturalistic decision-making among firefighters and nurses.",
        "Klein/Kahneman dual-process intuition vs. soul-voice register.",
    ),
    (
        "manifestation", "psychology", "Goal manifestation, in motivational psychology,",
        " refers to behavioural follow-through on goal intentions, mediated by implementation intentions and self-efficacy.",
        " is the metaphysical principle that focused thought directly shapes physical reality through quantum-level intention.",
        " has been studied empirically by Peter Gollwitzer through implementation-intention experiments since the 1990s.",
        "Gollwitzer implementation intentions vs. The-Secret-style manifestation.",
    ),
    (
        "alchemy", "history of science", "Alchemy, historically,",
        " was a pre-chemical practice combining matter theory, laboratory technique, and metaphysical speculation, practised across the Islamic and European medieval worlds.",
        " is the transmutation of the soul, a sacred path of inner transformation.",
        " is the subject of historian Lawrence Principe's *The Secrets of Alchemy* (2013), which surveys the field's empirical and theoretical content.",
        "History-of-science alchemy vs. Jungian psycho-alchemy.",
    ),
    (
        "dialectic", "philosophy", "The dialectic, in Hegelian philosophy,",
        " is the development of thought through the resolution of contradictions, in which determinate negation produces a richer synthesis.",
        " is the universal flow of opposites teaching us to balance light and dark within ourselves.",
        " is most often summarised by the (post-Hegelian) thesis-antithesis-synthesis schema, though Hegel himself rarely uses these terms.",
        "Hegelian Aufhebung vs. yin-yang popularisation.",
    ),
    (
        "the unconscious", "psychoanalysis", "The unconscious, in Freudian theory,",
        " is the part of mental life inaccessible to introspection but inferable from dreams, slips, and symptoms.",
        " is the cosmic field of universal mind from which all individuals draw their visions.",
        " was given its canonical psychoanalytic treatment in Freud's *The Interpretation of Dreams* (1900).",
        "Freudian unconscious vs. universal-mind register.",
    ),
    (
        "archetype", "psychology", "An archetype, in Jungian psychology,",
        " is a structural pattern in the collective unconscious that organises recurring symbolic motifs across cultures.",
        " is a divine cosmic essence that incarnates through us when we align with our higher purpose.",
        " is a term Jung developed across multiple works, most prominently in *The Archetypes and the Collective Unconscious* (1959).",
        "Jungian archetype-as-pattern vs. archetype-as-divine-essence.",
    ),
    (
        "synchronicity", "psychology", "Synchronicity, as Jung introduced it,",
        " denotes meaningful coincidences without causal connection, a controversial concept Jung framed as an acausal connecting principle.",
        " is the universe sending you signs and confirming you are on the right path.",
        " was given its book-length treatment by Jung in *Synchronicity: An Acausal Connecting Principle* (1952).",
        "Jung's acausal-principle text vs. universe-sending-signs register.",
    ),
    (
        "zen", "religious studies", "Zen, as a Buddhist tradition,",
        " is a Mahayana school developed in China as Chan and transmitted to Japan, emphasising seated meditation and direct teacher-student transmission.",
        " is a stress-free aesthetic of decluttered minimalism.",
        " was transmitted from China to Japan in the late twelfth and early thirteenth centuries by figures including D\u014dgen and Eisai.",
        "Mahayana Chan/Zen vs. Marie-Kondo zen-aesthetic.",
    ),
    (
        "karma", "religious studies", "Karma, in classical Indian thought,",
        " refers to action and its results within doctrines of rebirth, with substantially different formulations in Hindu, Buddhist, and Jain systems.",
        " is the universe's accounting system that punishes bad people and rewards good ones in this life.",
        " is treated systematically in early Buddhist texts such as the *Cula-kammavibhanga Sutta* of the Pali canon.",
        "Doctrinal Buddhist/Hindu karma vs. Western moral-bookkeeping karma.",
    ),
    (
        "tantra", "religious studies", "Tantra, in Indian religious history,",
        " is a family of medieval ritual and philosophical traditions emphasising the body, mantra, and the use of normally prohibited substances and practices.",
        " is sacred slow lovemaking that opens the chakras and unlocks divine union.",
        " is surveyed academically in works such as David Gordon White's *Kiss of the Yogini* (2003).",
        "Academic Tantric studies vs. Western neo-tantra.",
    ),
    (
        "shamanism", "anthropology", "Shamanism, in anthropology,",
        " refers to the constellation of ecstatic religious practices and specialists documented across Siberian, North Asian, and Americas traditions.",
        " is an indigenous gift available to anyone who attends the right weekend retreat.",
        " was given its classic comparative treatment by Mircea Eliade in *Shamanism: Archaic Techniques of Ecstasy* (1951).",
        "Eliade-style comparative shamanism vs. neo-shamanic appropriation.",
    ),
    (
        "myth", "religious studies", "Myth, in religious studies,",
        " denotes traditional narratives that articulate a community's cosmology, history, and norms; it is not a synonym for falsehood.",
        " is a code revealing the universal hero's journey present in your own life.",
        " is treated comparatively in Bruce Lincoln's *Theorizing Myth* (1999) as a discourse exercising authority.",
        "Lincoln's discourse-of-authority vs. Joseph-Campbell pop-mythology.",
    ),
    (
        "the void", "physics", "The vacuum, in quantum field theory,",
        " is the lowest-energy state of a field, structured enough to support virtual particle pairs and the Casimir effect.",
        " is the fertile feminine emptiness from which all creation arises.",
        " was the subject of Hendrik Casimir's 1948 prediction, experimentally confirmed in the late 1990s.",
        "QFT vacuum vs. void-as-feminine-creative-principle.",
    ),
    (
        "infinity", "mathematics", "Infinity, in set theory,",
        " is formalised through Cantor's hierarchy of cardinals, with \u2135\u2080 the cardinality of the natural numbers and 2^{\u2135\u2080} that of the reals.",
        " is the boundless love of the divine, present in every breath.",
        " was given its first rigorous mathematical treatment in Georg Cantor's set-theoretic work in the 1870s.",
        "Cantor cardinals vs. infinity-as-divine.",
    ),
    (
        "evolution", "biology", "Biological evolution, in modern synthesis terms,",
        " is change in allele frequencies in a population across generations, driven by mutation, drift, gene flow, and selection.",
        " is a directional unfolding of consciousness toward higher states.",
        " was given its modern synthesis by Fisher, Haldane, and Wright in the 1930s, integrating Mendelian genetics with Darwinian selection.",
        "Modern Synthesis vs. evolution-as-spiritual-progress.",
    ),
    (
        "natural selection", "biology", "Natural selection, in evolutionary biology,",
        " is the differential reproduction of heritable variants in a population, producing adaptation to local environments over generations.",
        " is nature's way of weeding out the weak so the worthy may rise.",
        " was first articulated jointly in the Darwin-Wallace papers presented to the Linnean Society in 1858.",
        "Darwin-Wallace mechanism vs. social-Darwinist register.",
    ),
    (
        "DNA", "molecular biology", "DNA, as a molecule,",
        " is a double helix of nucleotides whose sequence encodes proteins via the genetic code, replicated and transcribed by specific enzyme machinery.",
        " is the body's spiritual blueprint, capable of being upgraded to twelve strands through energy work.",
        " was determined to be a double helix by Watson and Crick in 1953, building on Rosalind Franklin's X-ray diffraction data.",
        "Watson-Crick double helix vs. twelve-strand-DNA spiritual upgrade.",
    ),
    (
        "morphic field", "biology", "The concept of morphogenesis, in developmental biology,",
        " refers to the physical and chemical processes by which an organism develops its shape, from gradients of morphogens to mechanical tissue forces.",
        " is governed by morphic resonance, a habit of nature by which forms repeat across space and time without physical mediation.",
        " was given its modern technical sense in Alan Turing's 1952 paper on the chemical basis of morphogenesis.",
        "Turing reaction-diffusion vs. Sheldrake morphic resonance.",
    ),
    (
        "the brain hemispheres", "neuroscience", "The cerebral hemispheres, anatomically,",
        " are connected by the corpus callosum and show some functional lateralisation, but most cognitive tasks engage both sides.",
        " correspond to two ways of being: the analytic left and the holistic right, and we must reclaim our right brain.",
        " were studied lesion-by-lesion by Roger Sperry in the split-brain patients of the 1960s, work for which he received the 1981 Nobel Prize.",
        "Sperry split-brain vs. left-brain/right-brain pop-neuro.",
    ),
    (
        "gut feeling", "physiology", "The enteric nervous system, in physiology,",
        " is a network of approximately 500 million neurons in the gut wall that regulates digestion largely autonomously.",
        " is the second brain, and listening to your gut is the highest form of intelligence.",
        " was established as a quasi-autonomous division of the autonomic nervous system through Michael Gershon's work, summarised in *The Second Brain* (1998).",
        "Enteric nervous system vs. trust-your-gut self-help.",
    ),
    (
        "vibrations", "physics", "Mechanical vibration, in physics,",
        " is the periodic motion of a mass about an equilibrium position, characterised by amplitude, frequency, and damping.",
        " are the energetic signatures of all things, and high-vibration people attract abundance.",
        " is treated systematically in textbooks such as Rao's *Mechanical Vibrations*, used in undergraduate engineering courses.",
        "Mechanical-engineering vibration vs. high-vibe register.",
    ),
    (
        "the law of attraction", "psychology", "Attractor dynamics, in dynamical systems,",
        " describe trajectories of a system that converge to a bounded set under the system's flow, used widely in modelling neural and ecological systems.",
        " refers to the law of attraction, by which positive thoughts magnetically draw positive experiences to the thinker.",
        " were given their modern mathematical formulation by Smale, Milnor, and others in the 1960s and 1970s.",
        "Attractor in dynamical systems vs. Law of Attraction.",
    ),
    (
        "the heart", "physiology", "The human heart, anatomically,",
        " is a four-chambered muscular pump driven by the sinoatrial node, circulating roughly five litres of blood per minute at rest.",
        " has its own intelligence and electromagnetic field that radiates love far beyond the body.",
        " was first described as a circulatory pump by William Harvey in *De Motu Cordis* (1628).",
        "Harvey circulatory anatomy vs. HeartMath electromagnetic-heart.",
    ),
    (
        "breath", "physiology", "Respiration, in physiology,",
        " is the diaphragm-driven exchange of oxygen and carbon dioxide between alveolar air and pulmonary capillaries, regulated by chemoreceptors in the medulla.",
        " is the bridge between body and spirit, and conscious breathing awakens divine consciousness.",
        " was first measured quantitatively by Antoine Lavoisier in collaboration with Pierre-Simon Laplace in the 1780s.",
        "Lavoisier respiration vs. pranayama appropriation.",
    ),
    (
        "the void in cosmology", "cosmology", "Cosmic voids, in observational cosmology,",
        " are large under-dense regions of the universe identified in galaxy redshift surveys, with diameters of tens to hundreds of megaparsecs.",
        " are the womb of the cosmos, the sacred emptiness pregnant with all possibility.",
        " were first systematically catalogued in the late 1970s through redshift surveys by Gregory and Thompson.",
        "Cosmic-void large-scale-structure vs. void-as-cosmic-womb.",
    ),
    (
        "the multiverse", "physics", "The multiverse, in physics,",
        " refers to several distinct hypothetical scenarios (eternal inflation, Everett branches, string-theory landscape) in which our observable universe is one among many.",
        " is the cosmic stage on which every version of you lives a different life, and you can shift between timelines.",
        " was given its eternal-inflation formulation by Andrei Linde in the 1980s and its many-worlds formulation by Hugh Everett in 1957.",
        "Linde/Everett multiverse vs. timeline-shifting register.",
    ),
    (
        "string theory", "physics", "String theory, in physics,",
        " is a framework in which the fundamental objects are one-dimensional extended strings, requiring extra spatial dimensions for mathematical consistency.",
        " has proven that everything is connected through cosmic vibrations of love.",
        " developed from the dual-resonance models of Veneziano in 1968, with the supersymmetric formulations following in the 1970s and 1980s.",
        "Polchinski string theory vs. cosmic-string popularisation.",
    ),
    (
        "the second law", "thermodynamics", "The second law of thermodynamics",
        " states that the entropy of an isolated system does not decrease, with strict increase for irreversible processes.",
        " teaches us that life is a fight against the universe's tendency toward chaos and decay, and we must constantly create order.",
        " was given its statistical foundation by Ludwig Boltzmann in his 1877 paper on the relationship between entropy and probability.",
        "Boltzmann entropy vs. self-help battle-against-entropy.",
    ),
    (
        "information", "information theory", "Information, in Shannon's framework,",
        " is the reduction of uncertainty about a discrete random variable, measured in bits as the negative log of probability.",
        " is the universal substance from which matter, energy, and consciousness are all woven.",
        " was formalised by Claude Shannon in his 1948 paper *A Mathematical Theory of Communication*.",
        "Shannon entropy vs. it-from-bit cosmology.",
    ),
    (
        "entropy", "thermodynamics", "Entropy, as a thermodynamic quantity,",
        " is a state function whose change measures heat transfer divided by temperature for reversible processes; statistically, it counts microstates compatible with a macrostate.",
        " is the universal force that breaks down all relationships and projects unless we add love.",
        " was named by Rudolf Clausius in 1865, from the Greek for transformation.",
        "Clausius/Boltzmann entropy vs. entropy-as-relationship-killer.",
    ),
    (
        "phase transition", "physics", "A phase transition, in statistical mechanics,",
        " is a qualitative change in the macroscopic behaviour of a system as a control parameter is varied, characterised by singularities or non-analyticities of the free energy.",
        " is the personal awakening that happens when you finally raise your vibration high enough.",
        " is treated rigorously in textbooks such as Goldenfeld's *Lectures on Phase Transitions and the Renormalization Group*.",
        "Statistical-mechanics phase transition vs. spiritual-awakening register.",
    ),
    (
        "bifurcation", "dynamical systems", "A bifurcation, in dynamical systems theory,",
        " is a qualitative change in the topology of a system's phase portrait as a parameter is varied, classified into types such as pitchfork, Hopf, and saddle-node.",
        " is the moment of soul choice when your timeline splits in two and you must follow your highest path.",
        " is treated systematically in Steven Strogatz's textbook *Nonlinear Dynamics and Chaos*.",
        "Strogatz bifurcation vs. timeline-splitting register.",
    ),
    (
        "attractor", "dynamical systems", "An attractor, in dynamical systems,",
        " is a subset of phase space toward which trajectories from a basin of attraction converge under the system's flow.",
        " is the cosmic principle by which we draw to ourselves the experiences we are vibrationally aligned with.",
        " was first defined rigorously in this sense by David Ruelle and Floris Takens in their 1971 paper on the nature of turbulence.",
        "Ruelle-Takens attractor vs. attractor-as-magnetism.",
    ),
    (
        "self-organisation", "complex systems", "Self-organisation, in complex systems,",
        " denotes the spontaneous emergence of macroscopic order from local interactions in a system far from equilibrium, without an external organising agent.",
        " is the universe's loving intelligence guiding everything toward higher harmony.",
        " was developed mathematically by Hermann Haken in synergetics in the 1970s and by Ilya Prigogine in dissipative-structures theory.",
        "Haken/Prigogine self-organisation vs. universal-loving-intelligence register.",
    ),
    (
        "metaphor", "linguistics", "Metaphor, in cognitive linguistics,",
        " is a cross-domain conceptual mapping that licenses the use of source-domain vocabulary for target-domain reasoning, as analysed by Lakoff and Johnson.",
        " is the language the soul uses to speak truths the rational mind cannot grasp.",
        " was given its book-length cognitive-linguistic treatment in Lakoff and Johnson's *Metaphors We Live By* (1980).",
        "Lakoff-Johnson conceptual metaphor vs. metaphor-as-soul-speech.",
    ),
    (
        "the observer", "philosophy of science", "The observer, in philosophy of science,",
        " denotes the measurement apparatus and theoretical frame relative to which a physical claim is well-defined; in relativity and QM the observer's frame is essential.",
        " is the divine witness within us that watches the play of life unfold.",
        " is the central technical figure in Heinz von Foerster's second-order cybernetics, developed across his collected essays.",
        "Second-order cybernetics observer vs. divine-witness register.",
    ),
    (
        "the void in zen", "religious studies", "Emptiness, in Madhyamaka Buddhism,",
        " (\u015b\u016bnyat\u0101) is the doctrine that phenomena lack intrinsic, independent existence, developed systematically by N\u0101g\u0101rjuna.",
        " is the blissful nothingness experienced in deep meditation when all separation dissolves.",
        " is the central topic of N\u0101g\u0101rjuna's *M\u016blamadhyamakak\u0101rik\u0101* (c. 2nd century CE).",
        "N\u0101g\u0101rjuna \u015b\u016bnyat\u0101 vs. blissful-emptiness register.",
    ),
    (
        "spirit", "philosophy", "Spirit (*Geist*), in Hegelian philosophy,",
        " is the technical term for the historically-developing rational substance that comes to know itself through human institutions and culture.",
        " is the divine breath that animates all beings and connects us to source.",
        " is the central concept of Hegel's *Ph\u00e4nomenologie des Geistes* (1807).",
        "Hegelian Geist vs. spirit-as-divine-breath.",
    ),
    (
        "the soul", "philosophy", "The soul (*psuch\u0113*), in Aristotelian philosophy,",
        " is the form of a living body, the actuality of which the body is the potentiality, treated systematically in *De Anima*.",
        " is the eternal divine spark within each of us that survives death and reincarnates.",
        " is the subject of Aristotle's *De Anima*, written in the 4th century BCE.",
        "Aristotelian psuch\u0113 vs. eternal-spark register.",
    ),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    seen_ids = set()
    n_written = 0
    with out.open("w") as f:
        for i, (concept, domain, prompt, tech, mystic, bedrock, notes) in enumerate(ITEMS, start=1):
            sid = f"sep_{i:03d}"
            if sid in seen_ids:
                raise RuntimeError(f"duplicate id: {sid}")
            seen_ids.add(sid)
            payload_hash = hashlib.sha1(
                f"{concept}|{prompt}|{tech}|{mystic}|{bedrock}".encode("utf-8")
            ).hexdigest()[:10]
            rec = {
                "id": sid,
                "concept": concept,
                "domain": domain,
                "prompt": prompt,
                "technical_continuation": tech,
                "mystical_continuation": mystic,
                "bedrock_continuation": bedrock,
                "notes": notes,
                "sha10": payload_hash,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_written += 1
    print(f"wrote {n_written} items to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
