"""Frozen, deterministic, tokenizer-agnostic text probes for Tycho Survey v3 Stage 1.

All probes are UTF-8 text tuples (context, target_a, target_b) or (ctx_a, ctx_b, target) so that
Pythia (GPT-NeoX BPE), OLMo-1 (Dolma BPE), and OLMo-2 (Dolma2 BPE) are evaluated on identical
semantic contrasts via exact character-aligned continuation bits-per-byte (bpb) and log-prob margins.
"""
from __future__ import annotations

import random
from dataclasses import dataclass


# 32 diverse, original expository passages (~110-150 words each) spanning science, history,
# computing, medicine, economics, astronomy, ecology, and law. Deterministic prefixes guarantee
# every passage easily exceeds 128 tokens across Pythia and OLMo tokenizers.
CORE_PASSAGES: list[str] = [
    (
        "During the late nineteenth century, coastal observatories began replacing manual tide gauges "
        "with mechanical drum recorders driven by float chambers connected to harbor basins. Because "
        "narrow intake pipes damped surface waves while passing slower tidal oscillations, engineers "
        "could extract harmonic components with millimeter repeatability over entire lunar cycles. "
        "Once telegraph networks linked distant ports, hydrographers compared phase lags across "
        "continental shelves and noticed that shallow estuaries distorted sinusoidal swells into "
        "steep incoming bores followed by prolonged ebbs. Those early continuous archives remain "
        "indispensable today, providing a century-scale baseline against which satellite altimetry "
        "calibrates modern estimates of regional sea-level rise and storm-surge amplification."
    ),
    (
        "In relational database engines, write-ahead logging separates the durability of committed "
        "transactions from the asynchronous flushing of dirty pages back to main table files. Before "
        "any modified B-tree node reaches disk, the engine appends a sequential record describing "
        "the physical or logical state change and forces that append buffer to stable storage. "
        "Sequential writes exploit high disk throughput while avoiding scattered random seeks during "
        "peak concurrency. If power fails unexpectedly, the recovery manager replays committed log "
        "entries forward from the most recent checkpoint and rolls back incomplete transactions using "
        "undo records, restoring exact ACID invariants without scanning every untouched data block."
    ),
    (
        "High-altitude conifers survive winter desiccation by coupling cellular sugar accumulation "
        "with gradual lignification of needle cuticles throughout autumn. As daylight shortens and "
        "nocturnal temperatures drop below freezing, trees synthesize raffinose family oligosaccharides "
        "that stabilize lipid membranes against ice-crystal shear. Simultaneously, bordered pits "
        "inside xylem tracheids act as microscopic check valves: when dissolved gases nucleate into "
        "embolism bubbles during freeze-thaw cycles, pressure differentials pull the central torus "
        "against the pit aperture, isolating the damaged conduit before air can propagate across "
        "neighboring water columns and collapse hydraulic transport when spring photosynthesis resumes."
    ),
    (
        "Classical Roman concrete owed much of its marine durability to reactive volcanic ash drawn "
        "from the Pozzuoli region near Naples. When mixed with slaked lime and seawater, aluminosilicate "
        "glass slowly dissolved under alkaline conditions and crystallized into interlocking plates of "
        "aluminum tobermorite and phillipsite along the mortar interface. Rather than weakening under "
        "constant wave action, microcracks admitted fresh seawater that reacted with residual quicklime "
        "clasts to precipitate new calcium carbonate seals. Modern materials scientists study harbor "
        "piers still standing after two millennia to design lower-carbon marine cements that cure at "
        "reduced kiln temperatures while resisting chloride corrosion."
    ),
    (
        "When an acoustic wave travels through a stratified ocean, variations in temperature, salinity, "
        "and hydrostatic pressure bend rays continuously toward regions of lower sound speed. Near the "
        "surface, warmer water raises sound velocity, whereas in the abyssal plain, rising pressure "
        "dominates and pushes velocity upward again. Between those two regimes lies the deep sound "
        "channel, roughly one kilometer below mid-latitude seas, where acoustic energy refracts back "
        "and forth without striking either the rough sea surface or the absorbing sediment floor. "
        "Because geometric spreading becomes cylindrical rather than spherical inside that waveguide, "
        "low-frequency whale vocalizations and oceanographic pings can remain detectable thousands of "
        "kilometers from their source."
    ),
    (
        "In modern compilers, static single assignment form simplifies dataflow optimization by "
        "requiring every virtual register to be defined at exactly one program point. Whenever two "
        "control-flow branches merge, the intermediate representation inserts a conceptual phi function "
        "that selects the appropriate incoming version according to the predecessor basic block. This "
        "explicit def-use graph makes constant propagation, dead-code elimination, and global value "
        "numbering run in nearly linear time without iterative bit-vector scans over every variable. "
        "During final instruction selection and register allocation, the backend lowers phi nodes into "
        "register copies along split edges and coalesces non-interfering lifetimes to minimize spills."
    ),
    (
        "Medieval European guild charters regulated far more than apprenticeship lengths and product "
        "prices; they served as municipal credit networks and quality-certification bodies in markets "
        "where formal contract enforcement was slow. A master weaver or silversmith posted a collective "
        "reputation bond backed by fellow members, allowing foreign merchants to purchase sealed bales "
        "without inspecting every bolt of cloth. Inspections occurred both at workshop looms and at "
        "city weighing halls, where defective dye lots incurred fines or public destruction. Although "
        "guilds eventually restricted entry and resisted mechanized looms, their assay marks established "
        "early institutional templates for standardized weights, trademarks, and commercial arbitration."
    ),
    (
        "Inside eukaryotic cells, the mitochondrial inner membrane folds into dense cristae that "
        "greatly expand surface area for oxidative phosphorylation. Electrons harvested from glucose "
        "and fatty acid oxidation pass through a chain of protein complexes that pump protons from the "
        "matrix into the intermembrane space, building both an electrical potential and a pH gradient. "
        "Protons flow back down this electrochemical gradient through ATP synthase, a rotary molecular "
        "turbine whose central stalk turns in discrete steps as catalytic subunits compress ADP and "
        "inorganic phosphate into ATP. Uncoupling proteins can intentionally short-circuit that proton "
        "flux in brown adipose tissue, releasing stored chemical energy directly as heat during cold exposure."
    ),
    (
        "Radio interferometry overcomes the diffraction limit of single parabolic dishes by combining "
        "phase-coherent signals recorded simultaneously at widely separated antennas. Each pair of "
        "telescopes samples a single spatial Fourier component of the sky brightness distribution, "
        "determined by the length and orientation of the projected baseline vector. As Earth rotates "
        "over several hours, those projected baselines sweep out elliptical tracks across the frequency "
        "plane, gradually filling in missing spatial frequencies through aperture synthesis. Atomic "
        "hydrogen maser clocks at continental stations stamp waveforms with picosecond stability so "
        "correlators can later align wavefronts offline and reconstruct sub-milliarcsecond images of "
        "galactic nuclei and protoplanetary disks."
    ),
    (
        "Suspension bridges carry roadway loads upward through vertical suspender cables into main "
        "parabolic cables anchored in massive concrete blocks at either shore. Under uniform dead load, "
        "a flexible cable naturally assumes a parabolic profile whose horizontal tension remains constant "
        "from tower to tower, transferring purely vertical compression into the tall pylons. Moving "
        "traffic and crosswinds, however, introduce asymmetric live loads and aerodynamic flutter that "
        "can excite torsional oscillations in slender decks. Following the 1940 collapse at Tacoma "
        "Narrows, structural engineers replaced solid plate girders with open stiffening trusses and "
        "streamlined box decks tested inside boundary-layer wind tunnels to shed vortices harmlessly."
    ),
    (
        "Economists distinguish between headline inflation, which tracks the full consumer basket "
        "including volatile food and energy contracts, and core inflation, which strips out short-lived "
        "supply shocks to expose persistent underlying price trends. Central banks watch service-sector "
        "wages and housing rents especially closely because multi-year lease structures and sticky labor "
        "agreements make those components slow to decelerate once expectations drift upward. When real "
        "interest rates rise above the neutral rate, borrowing slows first in residential construction "
        "and durable-goods manufacturing, whereas broad consumer prices respond only after several "
        "quarters of cooling demand narrow corporate pricing power across supply chains."
    ),
    (
        "Migratory songbirds integrate at least three distinct compass systems during nocturnal flights "
        "across continents. Before sunset, polarized light bands near the horizon calibrate a celestial "
        "heading that birds transfer to the rotating star pattern once darkness falls. On overcast "
        "nights, specialized cryptochrome proteins inside retinal cells appear to mediate radical-pair "
        "chemical reactions sensitive to the inclination angle of Earth's magnetic field lines relative "
        "to gravity. Because magnetic declination varies geographically, juveniles inherit a genetic "
        "vector program specifying flight duration and heading, whereas experienced adults refine that "
        "template with olfactory and landmark maps learned along previous stopover corridors."
    ),
    (
        "Error-correcting codes enable reliable digital communication over noisy radio and optical "
        "channels by mapping short message blocks into widely separated codewords in a higher-dimensional "
        "space. Low-density parity-check codes represent those constraints as sparse bipartite graphs "
        "linking variable nodes to parity-check nodes. During iterative belief propagation, each node "
        "exchanges log-likelihood messages summarizing how strongly neighboring constraints favor a zero "
        "or one bit, converging rapidly toward the maximum-likelihood codeword near the Shannon capacity "
        "limit. Modern wireless standards combine irregular LDPC matrices with hybrid automatic repeat "
        "requests so receivers can softly combine retransmitted parity bits without discarding noisy frames."
    ),
    (
        "Glacial varves form at the bottom of proglacial lakes where seasonal meltwater delivers "
        "alternating pulses of coarse silt and fine clay. During warm summer runoff, turbulent inflows "
        "carry pale sand and silt grains that settle rapidly onto the lakebed, whereas winter freeze-over "
        "halts sediment input and allows dark organic-rich clay particles to drift slowly out of "
        "suspension. Each light-dark couplet therefore records a single annual cycle, allowing geologists "
        "to count absolute calendar years directly from sediment cores much like dendrochronologists "
        "read tree rings. Cross-matching distinctive thick flood layers across neighboring basins lets "
        "paleoclimatologists reconstruct ice-sheet retreat rates to within a few decades."
    ),
    (
        "Public-key cryptography revolutionized network security by replacing pre-shared secret keys "
        "with asymmetric mathematical trapdoors that anyone can evaluate forward yet practically nobody "
        "can invert without private auxiliary information. Elliptic-curve protocols build those one-way "
        "groups over finite fields where scalar point multiplication is fast via double-and-add ladders "
        "while the inverse discrete logarithm scales sub-exponentially only for weak curves. Because "
        "shorter key lengths match the classical security of much larger integer-factorization moduli, "
        "elliptic-curve handshakes sharply reduce bandwidth and latency during TLS session setup, "
        "though standardization bodies are now deploying lattice-based key encapsulation to prepare "
        "for future fault-tolerant quantum computers."
    ),
    (
        "Japanese woodblock printmaking during the Edo period operated as a tightly coordinated "
        "publishing industry divided among four specialized workshops. A publisher commissioned an artist "
        "to draw ink keylines on thin translucent paper, which a master carver pasted face-down onto "
        "seasoned cherrywood blocks and cut away until only raised ridges remained. Separate color "
        "blocks were carved from proof sheets using tiny L-shaped registration notches called kento cut "
        "into the corners of every plank. Printers brushed water-based mineral and botanical pigments "
        "onto each block and rubbed damp mulberry paper by hand with a bamboo-sheath baren, achieving "
        "subtle tonal gradations impossible on early European metal presses."
    ),
    (
        "In superconducting quantum circuits, Josephson junctions replace ordinary linear inductors "
        "with nonlinear tunneling barriers formed by a nanometer-thick oxide layer separating two "
        "aluminum films. That nonlinearity creates unevenly spaced energy levels, enabling microwave "
        "pulses to drive transitions exclusively between the ground and first excited states without "
        "leaking population into higher rungs. To protect delicate superpositions from thermal photons "
        "and electromagnetic noise, chips sit inside multi-stage dilution refrigerators cooled to ten "
        "millikelvin and shielded by concentric magnetic cans. Dispersive readout resonators weakly "
        "coupled to each transmon shift slightly in frequency depending on qubit state, allowing fast "
        "quantum non-demolition measurement."
    ),
    (
        "Coral reefs thrive in nutrient-poor tropical waters because reef-building polyps maintain an "
        "intimate metabolic partnership with photosynthetic dinoflagellates housed inside their tissue "
        "cells. The algae convert sunlight and dissolved carbon dioxide into glycerol and amino acids "
        "that supply up to ninety percent of the coral host's daily energy budget, while the polyp "
        "returns nitrogenous waste and a sheltered calcifying skeleton. Sustained marine heatwaves "
        "disrupt photosystem electron transport inside the symbionts, generating reactive oxygen species "
        "that force the host to expel its algae and turn stark white. If thermal stress subsides "
        "quickly, colonies can reacquire heat-tolerant algal strains, though repeated back-to-back "
        "bleaching events exhaust lipid reserves and trigger widespread mortality."
    ),
    (
        "Automatic speech recognition shifted decisively from modular hidden Markov pipelines toward "
        "end-to-end neural sequence transducers over the past decade. Traditional systems trained "
        "separate acoustic models, pronunciation lexicons, and n-gram language models, requiring hand-"
        "engineered phonetic alignments whenever new dialects or technical terms entered deployment. "
        "Modern conformer encoders combine convolutional front-ends that capture fine local spectral "
        "formants with self-attention layers that resolve long-range speaker adaptation and syntactic "
        "context across entire utterances. Paired with subword tokenizers and streaming transducer "
        "decoders, these architectures emit transcriptions with low latency even amid reverberant "
        "rooms, overlapping voices, and rapid code-switching between languages."
    ),
    (
        "Deep-sea hydrothermal vents support chemosynthetic ecosystems completely decoupled from "
        "surface sunlight. Cold seawater percolates downward through fractured basaltic crust near "
        "mid-ocean ridges, heats above three hundred degrees Celsius near shallow magma chambers, and "
        "leaches dissolved sulfides, iron, and manganese before shooting back through mineral chimneys. "
        "Specialized bacteria oxidize hydrogen sulfide to fix dissolved carbon into organic sugars, "
        "forming dense microbial mats grazed by blind shrimp and hosting symbiotic colonies inside the "
        "plumes of giant tube worms. Because individual vent fields remain active for only decades "
        "before tectonic shifts seal their plumbing, vent larvae must disperse hundreds of kilometers "
        "along abyssal currents to colonize newly erupted fissures."
    ),
    (
        "Maritime insurance in seventeenth-century London evolved inside coffeehouses where merchants "
        "and shipowners circulated handwritten shipping lists detailing vessel tonnage, hull age, and "
        "recent storm reports. Underwriters wrote their names beneath proposed policy terms along with "
        "the fraction of cargo value they agreed to cover, spreading catastrophic shipwreck risk across "
        "dozens of independent balance sheets. Over time, pooled loss ledgers revealed statistical "
        "patterns linking winter North Atlantic crossings, wooden hull sheathing, and captain experience "
        "to claims frequency. Those empirical actuarial tables not only stabilized overseas trade finance "
        "but also drove shipbuilders to adopt stricter classification surveys and safer load-line markings."
    ),
    (
        "Lithium-ion battery degradation reflects a tangled interplay of chemical film growth and "
        "mechanical stress rather than a single failure mode. During the initial formation cycles, "
        "electrolyte decomposes on the graphite anode to create a thin solid-electrolyte interphase "
        "that conducts lithium ions while blocking further electron transfer. Repeated fast charging "
        "at cold temperatures, however, can drive metallic lithium plating onto anode surfaces, "
        "permanently consuming cyclable inventory and risking internal short circuits through needle-"
        "like dendrites. Simultaneously, volume expansion and contraction inside layered nickel-rich "
        "cathode particles open microscopic cracks that expose fresh surfaces to side reactions, "
        "gradually raising internal impedance and trimming usable driving range."
    ),
    (
        "In civil litigation, appellate courts rarely retry witness credibility or reweigh disputed "
        "factual testimony; instead, they review the written trial record for reversible errors of law "
        "or procedural abuse. Trial judges enjoy broad deference on evidentiary rulings because they "
        "observe live demeanor and manage courtroom context directly, whereas statutory interpretation "
        "and constitutional questions receive fresh de novo scrutiny by multi-judge panels. Written "
        "appellate opinions serve a dual institutional role: resolving the concrete dispute between "
        "the parties while articulating general principles that guide lower courts and private actors "
        "in future transactions. When intermediate circuits split over the meaning of a federal statute, "
        "the highest court often intervenes to restore uniform national doctrine."
    ),
    (
        "Modern weather forecasting relies on four-dimensional variational data assimilation to blend "
        "millions of noisy atmospheric observations with numerical fluid-dynamics models every six "
        "hours. Raw satellite radiances, commercial aircraft wind reports, ocean buoys, and radiosonde "
        "balloons arrive asynchronously at irregular locations, so assimilation algorithms solve a large "
        "inverse optimization problem finding the physically consistent initial state that best fits "
        "both recent measurements and short-range model trajectories. Because chaotic convection amplifies "
        "tiny initialization uncertainties exponentially over days, meteorological centers run ensembles "
        "of perturbed simulations to map forecast probabilities rather than relying on a single "
        "deterministic track when warning coastal communities about approaching cyclones."
    ),
    (
        "Fermented sourdough breads develop their complex acidity and open crumb structure through a "
        "stable symbiosis between wild yeasts and heterofermentative lactic acid bacteria. Unlike "
        "commercial baker's yeast bred purely for rapid carbon dioxide production, sourdough yeasts "
        "tolerate low pH environments created as lactobacilli convert flour maltose into lactic and "
        "acetic acids. Cooler bulk fermentation tilts metabolism toward sharper acetic notes and slower "
        "gluten relaxation, while warmer temperatures favor milder creamy lactic profiles. Beyond flavor, "
        "acidification activates endogenous cereal phytases that break down mineral-binding phytic acid "
        "and slows starch retrogradation after baking, keeping loaves tender and resistant to spoilage "
        "for days without synthetic preservatives."
    ),
    (
        "Satellite navigation receivers determine three-dimensional position and precise local clock "
        "offset by measuring signal arrival times from at least four orbiting transmitters simultaneously. "
        "Each satellite broadcasts ephemeris parameters describing its exact orbital trajectory along "
        "with timestamps steered by onboard rubidium or cesium atomic clocks. Because radio waves "
        "travel roughly thirty centimeters per nanosecond, even microsecond clock errors would produce "
        "hundreds of meters of positioning drift if relativistic effects were ignored: satellite clocks "
        "run slightly slower due to orbital velocity under special relativity yet noticeably faster "
        "due to weaker gravity under general relativity, requiring a continuous net adjustment of "
        "thirty-eight microseconds per day to keep global navigation accurate."
    ),
    (
        "Industrial desalination plants increasingly rely on reverse osmosis membranes rather than "
        "thermal distillation because pushing pressurized seawater through semipermeable polymer films "
        "avoids the steep latent heat cost of boiling water. Spiral-wound polyamide sheets reject over "
        "ninety-nine percent of dissolved sodium and chloride ions while letting pure water molecules "
        "diffuse across at operating pressures near sixty bars. Energy recovery exchangers capture "
        "hydraulic pressure from the rejected brine stream and transfer it directly back to incoming "
        "feedwater, cutting electricity consumption by more than half compared to early installations. "
        "Operators still monitor biological fouling and mineral scaling closely, balancing pretreatment "
        "ultrafiltration with environmentally responsible diffuser arrays that dilute discharged brine."
    ),
    (
        "Print cartographers of the sixteenth century transformed scattered sailors' portolan charts "
        "into unified world projections designed specifically for open-ocean compass navigation. Gerardus "
        "Mercator mathematically stretched latitude spacing progressively toward the poles so that any "
        "line of constant compass bearing—known as a rhumb line—appeared as a straight ruler segment "
        "on the flat chart. Although that conformal geometry badly inflated the apparent land area of "
        "Greenland and Antarctica relative to equatorial continents, navigators accepted the areal "
        "distortion gladly because plotting courses no longer required tedious spherical trigonometry "
        "on pitching wooden decks. Digital web maps later revived the exact same cylindrical projection "
        "to preserve clean ninety-degree street intersections at every zoom level."
    ),
    (
        "In semiconductor photolithography, printing transistor features smaller than fifteen nanometers "
        "requires extreme ultraviolet light at a wavelength of thirteen and a half nanometers generated "
        "by blasting microscopic tin droplets with high-power carbon-dioxide lasers inside vacuum "
        "chambers. Because glass lenses absorb extreme ultraviolet photons almost completely, optical "
        "columns replace refractive elements with ultra-flat multilayer molybdenum-silicon mirrors "
        "engineered to atomic smoothness so constructive Bragg reflection redirects roughly seventy "
        "percent of incident energy per bounce. Every wafer stage moves with sub-nanometer interferometric "
        "alignment across hundreds of exposure fields per hour, turning plasma physics, precision "
        "mechatronics, and stochastic photoresist chemistry into a tightly synchronized mass-manufacturing pipeline."
    ),
    (
        "Urban heat islands arise when dark asphalt roadways, dense masonry facades, and sparse "
        "vegetative canopies absorb solar radiation throughout the afternoon and release stored "
        "sensible heat slowly after sunset. Whereas suburban soils and parks dissipate a large share "
        "of incoming energy through evaporative cooling, impervious metropolitan cores retain heat "
        "inside canyon-like street geometries that block nighttime radiative escape to the open sky "
        "and stifle cooling breezes. Elevated nocturnal temperatures carry the sharpest public-health "
        "burden during summer heatwaves because human cardiovascular systems lose their overnight "
        "recovery window, prompting municipal planners to combine high-albedo roof coatings, shaded "
        "street-tree corridors, and permeable pavements to flatten peak nighttime anomalies."
    ),
    (
        "Acoustic guitars amplify vibrating strings without electronics by coupling bridge tension to "
        "a thin spruce soundboard stiffened underneath with sculpted wooden X-bracing. Plucked strings "
        "alone move too little air to project across a room, so downward pressure at the saddle rocks "
        "the top plate like a flexible piston whose resonant modes radiate sound waves efficiently "
        "across several octaves. Meanwhile, air trapped inside the hollow body oscillates through the "
        "central soundhole as a Helmholtz resonator that reinforces the lowest bass frequencies. "
        "Luthiers tap-tune individual braces by shaving fractions of a millimeter off spruce ribs, "
        "balancing structural resilience against seventy kilograms of string pull with brisk harmonic "
        "responsiveness and balanced sustain."
    ),
    (
        "Adaptive optics systems installed on mountain-top telescopes restore diffraction-limited "
        "clarity to ground-based astronomy by measuring and canceling atmospheric turbulence a thousand "
        "times per second. As starlight descends through shifting thermal air pockets, its originally "
        "flat wavefront wrinkles into rapidly boiling speckles that blur fine planetary and stellar "
        "detail. A wavefront sensor tracks either a bright nearby guide star or an artificial sodium "
        "beacon excited ninety kilometers overhead by a yellow laser, computing real-time phase errors "
        "across hundreds of sub-apertures. High-speed actuators behind a thin deformable mirror flex "
        "its reflective surface by nanometers in opposite phase, flattening incoming light before it "
        "reaches infrared spectrographs and coronagraphs hunting faint exoplanets."
    ),
]


@dataclass(frozen=True)
class PairProbeItem:
    """Two contexts predicting the same target, OR one context comparing target_a vs target_b."""
    kind: str           # "ctx_pair" or "tgt_pair" or "dual_tgt_pair"
    ctx_a: str
    ctx_b: str          # used when kind in ("ctx_pair", "dual_tgt_pair")
    target_a: str       # primary/good/clean target
    target_b: str       # contrast/bad/decoy target (empty when kind == "ctx_pair")


def _split_clause(passage: str) -> tuple[str, str]:
    """Split a passage into (prefix, final_sentence_with_leading_space)."""
    s = passage.strip()
    idx = s.rfind(". ", 0, len(s) - 20)
    if idx < 0:
        words = s.split()
        return " ".join(words[:-14]), " " + " ".join(words[-14:])
    return s[: idx + 1], s[idx + 1 :]


def _shuffle_windows(text: str, win: int, rng: random.Random) -> str:
    words = text.split()
    out: list[str] = []
    for i in range(0, len(words), win):
        chunk = words[i : i + win]
        if len(chunk) > 2:
            rng.shuffle(chunk)
        out.extend(chunk)
    return " ".join(out)


def build_core_passages(n_seqs: int = 32) -> list[str]:
    """Return `n_seqs` deterministic core passages (prefixes of the frozen bank)."""
    if n_seqs <= len(CORE_PASSAGES):
        return list(CORE_PASSAGES[:n_seqs])
    # Deterministically pair distinct passages if more than 32 requested
    out = list(CORE_PASSAGES)
    i = 0
    while len(out) < n_seqs:
        a = CORE_PASSAGES[i % len(CORE_PASSAGES)]
        b = CORE_PASSAGES[(i * 7 + 3) % len(CORE_PASSAGES)]
        out.append(a + " " + b)
        i += 1
    return out


def build_channel3_probes(seed: int = 20261127, max_per_probe: int = 32) -> dict[str, list[PairProbeItem]]:
    """Build all 8 tokenizer-agnostic matched probe pairs for Channel 3."""
    rng = random.Random(seed)
    probes: dict[str, list[PairProbeItem]] = {}

    # P1_order: clean passage prefix vs window-shuffled prefix predicting exact final sentence
    p1: list[PairProbeItem] = []
    for p in CORE_PASSAGES[:max_per_probe]:
        prefix, target = _split_clause(p)
        shuf = _shuffle_windows(prefix, win=10, rng=rng)
        p1.append(PairProbeItem("ctx_pair", prefix, shuf, target, ""))
    probes["P1_order"] = p1

    # P2_horizon: full prefix (~90 words) vs truncated 12-word suffix predicting final sentence
    p2: list[PairProbeItem] = []
    for p in CORE_PASSAGES[:max_per_probe]:
        prefix, target = _split_clause(p)
        short = " ".join(prefix.split()[-12:])
        p2.append(PairProbeItem("ctx_pair", prefix, short, target, ""))
    probes["P2_horizon"] = p2

    # P3_copy_decoy: key-value retrieval with a near-match decoy distractor
    keys = [
        "amber", "cobalt", "quartz", "cedar", "maple", "granite", "velvet", "copper",
        "silver", "indigo", "basalt", "linen", "nickel", "topaz", "willow", "garnet",
        "slate", "birch", "marble", "ivory", "bronze", "coral", "flint", "walnut",
        "jasper", "pewter", "alder", "ochre", "saffron", "myrtle", "shale", "nickel",
    ]
    vals = [
        "harbor", "canyon", "meadow", "glacier", "orchard", "beacon", "cavern", "summit",
        "lagoon", "prairie", "citadel", "valley", "terrace", "plateau", "harbor", "basin",
        "ridge", "estuary", "island", "garden", "bridge", "tower", "anchor", "compass",
        "lantern", "feather", "pebble", "cinder", "stream", "timber", "boulder", "crater",
    ]
    p3: list[PairProbeItem] = []
    for i in range(max_per_probe):
        k_sample = rng.sample(list(dict.fromkeys(keys)), 5)
        v_sample = rng.sample(list(dict.fromkeys(vals)), 5)
        k_tgt, v_tgt = k_sample[0], v_sample[0]
        v_dec = v_sample[1]
        # Clean: unambiguous table
        rows_clean = [f"{k_sample[j]} -> {v_sample[j]}" for j in range(4)]
        rng.shuffle(rows_clean)
        ctx_clean = "Lookup table: " + "; ".join(rows_clean) + f". Query: {k_tgt} ->"
        # Decoy: repeats k_tgt in a distractor note pointing to v_dec right after the table
        rows_dec = [f"{k_sample[j]} -> {v_sample[j]}" for j in range(4)]
        rng.shuffle(rows_dec)
        ctx_dec = (
            "Lookup table: "
            + "; ".join(rows_dec)
            + f". Unrelated memo: not {k_tgt} -> {v_dec}, keep original table. Query: {k_tgt} ->"
        )
        p3.append(PairProbeItem("dual_tgt_pair", ctx_clean, ctx_dec, " " + v_tgt, " " + v_dec))
    probes["P3_copy_decoy"] = p3

    # P4_induction_vs_break: repeated 6-word list continuation (repeat target vs non-repeat foil)
    nouns = [
        "river", "clock", "stone", "chair", "train", "cloud", "glass", "wheel",
        "plant", "paper", "shell", "flame", "bread", "crown", "piano", "field",
        "house", "sword", "apple", "storm", "knife", "shirt", "beach", "horse",
        "brick", "table", "grape", "light", "brush", "penny", "tower", "river",
    ]
    uniq_nouns = list(dict.fromkeys(nouns))
    p4: list[PairProbeItem] = []
    for i in range(max_per_probe):
        ws = rng.sample(uniq_nouns, 8)
        seq = " ".join(ws[:6])
        ctx = f"Sequence: {seq}. Repeat: {' '.join(ws[:5])}"
        # target_a = exact induction token ws[5]; target_b = novel non-repeated foil ws[6]
        p4.append(PairProbeItem("tgt_pair", ctx, "", " " + ws[5], " " + ws[6]))
    probes["P4_induction_vs_break"] = p4

    # P5_agreement_attractor: subject-verb agreement simple vs intervening plural/singular attractor
    agr_templates = [
        ("The key to the cabinet", "The key to the old wooden cabinets", " is", " are"),
        ("The author of the letter", "The author of the famous historical letters", " was", " were"),
        ("The manager of the store", "The manager of the three downtown stores", " has", " have"),
        ("The roof of the house", "The roof of the newly painted houses", " is", " are"),
        ("The driver of the truck", "The driver of the heavy delivery trucks", " was", " were"),
        ("The owner of the boat", "The owner of the bright yellow boats", " has", " have"),
        ("The color of the flower", "The color of the tropical garden flowers", " is", " are"),
        ("The price of the ticket", "The price of the front-row concert tickets", " was", " were"),
        ("The captain of the ship", "The captain of the large cargo ships", " has", " have"),
        ("The flavor of the cake", "The flavor of the freshly baked cakes", " is", " are"),
        ("The sound of the bell", "The sound of the distant cathedral bells", " was", " were"),
        ("The leader of the team", "The leader of the regional research teams", " has", " have"),
        ("The keys to the cabinets", "The keys to the old wooden cabinet", " are", " is"),
        ("The authors of the letters", "The authors of the famous historical letter", " were", " was"),
        ("The managers of the stores", "The managers of the large downtown store", " have", " has"),
        ("The roofs of the houses", "The roofs of the newly painted house", " are", " is"),
        ("The drivers of the trucks", "The drivers of the heavy delivery truck", " were", " was"),
        ("The owners of the boats", "The owners of the bright yellow boat", " have", " has"),
        ("The colors of the flowers", "The colors of the tropical garden flower", " are", " is"),
        ("The prices of the tickets", "The prices of the front-row concert ticket", " were", " was"),
        ("The captains of the ships", "The captains of the large cargo ship", " have", " has"),
        ("The flavors of the cakes", "The flavors of the freshly baked cake", " are", " is"),
        ("The sounds of the bells", "The sounds of the distant cathedral bell", " were", " was"),
        ("The leaders of the teams", "The leaders of the regional research team", " have", " has"),
        ("The bridge across the river", "The bridge across the winding mountain rivers", " is", " are"),
        ("The bridges across the rivers", "The bridges across the winding mountain river", " are", " is"),
        ("The painter of the mural", "The painter of the colorful street murals", " was", " were"),
        ("The painters of the murals", "The painters of the colorful street mural", " were", " was"),
        ("The student in the class", "The student in the advanced physics classes", " has", " have"),
        ("The students in the classes", "The students in the advanced physics class", " have", " has"),
        ("The lock on the gate", "The lock on the heavy iron gates", " is", " are"),
        ("The locks on the gates", "The locks on the heavy iron gate", " are", " is"),
    ]
    p5 = [
        PairProbeItem("dual_tgt_pair", s_ctx, a_ctx, good, bad)
        for (s_ctx, a_ctx, good, bad) in agr_templates[:max_per_probe]
    ]
    probes["P5_agreement_attractor"] = p5

    # P6_memory_vs_icl: parametric fact vs explicit in-context counterfactual override
    facts = [
        ("the capital of France", "Paris", "Lyon"),
        ("the capital of Japan", "Tokyo", "Osaka"),
        ("the capital of Italy", "Rome", "Milan"),
        ("the capital of Germany", "Berlin", "Munich"),
        ("the capital of Spain", "Madrid", "Seville"),
        ("the capital of Egypt", "Cairo", "Luxor"),
        ("the capital of Canada", "Ottawa", "Toronto"),
        ("the capital of Ireland", "Dublin", "Cork"),
        ("the capital of Belgium", "Brussels", "Ghent"),
        ("the capital of Russia", "Moscow", "Sochi"),
        ("the capital of Greece", "Athens", "Sparta"),
        ("the capital of Sweden", "Stockholm", "Malmo"),
        ("the capital of Norway", "Oslo", "Bergen"),
        ("the capital of Poland", "Warsaw", "Gdansk"),
        ("the capital of Portugal", "Lisbon", "Porto"),
        ("the capital of Austria", "Vienna", "Graz"),
        ("the chemical symbol for gold", "Au", "Ag"),
        ("the chemical symbol for iron", "Fe", "Cu"),
        ("the chemical symbol for silver", "Ag", "Zn"),
        ("the largest planet in the solar system", "Jupiter", "Mars"),
        ("the planet closest to the Sun", "Mercury", "Venus"),
        ("the author of Hamlet", "Shakespeare", "Dickens"),
        ("the continent containing Kenya", "Africa", "Asia"),
        ("the continent containing France", "Europe", "Asia"),
        ("the ocean west of California", "Pacific", "Atlantic"),
        ("the language primarily spoken in Mexico", "Spanish", "French"),
        ("the currency used in Japan", "yen", "euro"),
        ("the boiling point of water in Celsius", "100", "50"),
        ("the number of days in a week", "seven", "nine"),
        ("the number of months in a year", "twelve", "ten"),
        ("the season following winter", "spring", "autumn"),
        ("the metal attracted by a magnet", "iron", "copper"),
    ]
    p6: list[PairProbeItem] = []
    for subj, true_ans, cf_ans in facts[:max_per_probe]:
        ctx_clean = f"Question: What is {subj}? Answer:"
        ctx_cf = (
            f"Story: In this fictional world, {subj} is {cf_ans}. "
            f"Question: In this fictional world, what is {subj}? Answer:"
        )
        # Note: target_a = true_ans (parametric), target_b = cf_ans (counterfactual ICL)
        # Clean margin = logP(true) - logP(cf); CF override margin = logP(cf) - logP(true) = -(margin_b)!
        p6.append(PairProbeItem("dual_tgt_pair", ctx_clean, ctx_cf, " " + true_ans, " " + cf_ans))
    probes["P6_memory_vs_icl"] = p6

    # P7_binding_2_vs_5: 2-entity vs 5-entity variable binding
    names = ["Alice", "Bob", "Carol", "David", "Elena", "Frank", "Grace", "Henry", "Irene", "Jacob"]
    cities = ["Rome", "Oslo", "Lima", "Bern", "Riga", "Lyon", "Cork", "Bath", "York", "Ghent"]
    p7: list[PairProbeItem] = []
    for i in range(max_per_probe):
        ns = rng.sample(names, 5)
        cs = rng.sample(cities, 5)
        q_idx = rng.randrange(2)  # query one of the first 2 so both 2-ent and 5-ent contain it
        other_idx = 1 - q_idx
        ctx2 = f"{ns[0]} lives in {cs[0]}. {ns[1]} lives in {cs[1]}. Question: Where does {ns[q_idx]} live? Answer:"
        sents5 = [f"{ns[j]} lives in {cs[j]}." for j in range(5)]
        rng.shuffle(sents5)
        ctx5 = " ".join(sents5) + f" Question: Where does {ns[q_idx]} live? Answer:"
        p7.append(PairProbeItem("dual_tgt_pair", ctx2, ctx5, " " + cs[q_idx], " " + cs[other_idx]))
    probes["P7_binding_2_vs_5"] = p7

    # P8_arith_carry: 2-digit addition no-carry vs with-carry (comparing exact answer vs buggy carry-dropped foil)
    p8: list[PairProbeItem] = []
    for i in range(max_per_probe):
        # No-carry pair: a1+b1 <= 8, a0+b0 <= 9
        a1, b1 = rng.randint(1, 4), rng.randint(1, 4)
        a0, b0 = rng.randint(1, 4), rng.randint(1, 5)
        na, nb = 10 * a1 + a0, 10 * b1 + b0
        nsum = na + nb
        # With-carry pair: c0+d0 >= 11, c1+d1 <= 7
        c1, d1 = rng.randint(1, 3), rng.randint(1, 4)
        c0, d0 = rng.randint(6, 9), rng.randint(5, 9)
        ca, cb = 10 * c1 + c0, 10 * d1 + d0
        csum = ca + cb
        # Contrast: compare no-carry sum vs carry sum (both evaluated on exact continuation bpb;
        # foil for ctx_a is nsum+10, foil for ctx_b is csum-10 (dropped carry!)).
        # To keep PairProbeItem uniform, encode ctx_a=no-carry, ctx_b=with-carry via separate items?
        # Even cleaner: store exact answer in target_a (for ctx_a) and exact carry answer in target_b (for ctx_b),
        # tagged with kind="arith_pair"!
        p8.append(
            PairProbeItem(
                "arith_pair",
                f"Compute: {na} + {nb} =",
                f"Compute: {ca} + {cb} =",
                f" {nsum}",
                f" {csum}",
            )
        )
    probes["P8_arith_carry"] = p8

    return probes


def build_gram_prompts(n_prompts: int = 128, seed: int = 20261127) -> list[str]:
    """Return `n_prompts` deterministic prompt strings ending on diverse content words.

    Avoids repeating identical terminal punctuation ('.', '->', ':') across prompts, which
    would otherwise create a trivial rank-1 token-embedding cluster in K_l at early layers.
    """
    prompts: list[str] = []
    seen_last_words: set[str] = set()
    # Slice each of the 32 core passages at 4 distinct interior word boundaries
    for frac in (0.30, 0.50, 0.70, 0.88):
        for p in CORE_PASSAGES:
            words = p.split()
            idx = max(12, min(len(words) - 4, int(len(words) * frac)))
            # Nudge idx locally to land on a clean alphabetic word not yet repeated if possible
            chosen = idx
            for off in (0, 1, -1, 2, -2, 3, -3):
                j = idx + off
                if 10 <= j < len(words) - 2:
                    w = words[j - 1].strip(",.;:()\"'").lower()
                    if w.isalpha() and len(w) >= 3 and w not in seen_last_words:
                        chosen = j
                        seen_last_words.add(w)
                        break
            frag = " ".join(words[:chosen]).rstrip(",.;:()\"'")
            prompts.append(frag)
            if len(prompts) >= n_prompts:
                return prompts
    return prompts[:n_prompts]
