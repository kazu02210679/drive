# Design QA — Architecture Map

## Evidence

- Source visual truth: `../reference-x-tweet.png` (X post viewport capture). The architecture-video region used for focused comparison is `x=216, y=286, 560×341`.
- Journey-structure reference: [`joeyvansommeren/journey-mapper`](https://github.com/joeyvansommeren/journey-mapper), linked from the second X post. Applied concepts: scenario stages, service-blueprint backstage detail, factual evidence, and explicit coverage gaps.
- Implementation screenshot: `docs/qa/implementation-flow-cards-final.png`.
- Responsive screenshot: `docs/qa/implementation-map-mobile.png`.
- First comparison: `docs/qa/design-qa-comparison-pass1.png`.
- Final comparison: `docs/qa/design-qa-comparison-pass2.png`.
- Flow-inspector comparison: `docs/qa/design-qa-comparison-pass3.png`.
- Flow-navigation implementation: `docs/qa/implementation-flow-navigation-final.png`.
- Mobile auto-scroll implementation: `docs/qa/implementation-flow-mobile-auto-scroll.png`.
- Final Flow-navigation comparison: `docs/qa/design-qa-comparison-pass4.png`.
- Large Flow-card implementation: `docs/qa/implementation-flow-cards-final.png`.
- Mobile Flow-card implementation: `docs/qa/implementation-flow-cards-mobile-options.png`.
- Mobile auto-scroll implementation after the latest change: `docs/qa/implementation-flow-cards-mobile.png`.
- Latest comparison: `docs/qa/design-qa-comparison-pass5.png`.
- Browser viewport: `1280×720` CSS px, `devicePixelRatio=1` for desktop; `390×844` CSS px, `devicePixelRatio=1` for mobile.
- Source raster: `1265×712` px. Desktop implementation raster: `1280×720` px. Mobile implementation rasters: `390×844` px.
- Focused comparison normalization: the `560×341` source media crop was proportionally scaled and padded to `1280×720`; the desktop implementation was already `1280×720`; both were combined into a `2560×720` side-by-side image.
- State: desktop dark theme, `すべての関係` selected with all four Flow cards visible, no node selection or search query. Mobile evidence shows `Evidence & evaluation` selected, both at the automatic detail-scroll destination and after scrolling back to the four choices.
- Focused comparison: the right-side Flow cards and inspector are large enough in `design-qa-comparison-pass5.png` to judge their one-column layout, descriptions, selected state, title hierarchy, and relationship to the graph; no additional crop was needed.

## Findings

No actionable P0, P1, or P2 mismatch remains.

- Fonts and typography: the source uses compact white sans-serif labels over a dark technical canvas. The implementation uses a system sans-serif with comparable compact weights, clear hierarchy, readable inspector copy, controlled line wrapping, and monospace treatment for source paths. The larger Japanese product title is an intentional localization, not a fidelity defect.
- Spacing and layout rhythm: the graph remains the dominant region, with a persistent right inspector close to the source proportion. Removing the redundant node picker and top Flow selector leaves the toolbar quieter. The node-color legend sits directly under the map description, before the graph. Flow choices sit above the selected Flow content as four large cards in a 1×4 stack, with 8px gaps and 64px minimum tap targets. The cards participate in normal inspector flow so automatic detail scrolling does not trap the user behind a sticky selector; scrolling upward reveals the complete choice set. The workspace is fixed to the viewport on desktop so long inspector content scrolls independently; on mobile, the same content becomes one vertical document without horizontal overflow.
- Colors and visual tokens: near-black surfaces, restrained borders, amber active paths, dim inactive structure, and category-colored outlines preserve the source's visual logic. Text and interactive controls retain clear contrast.
- Image quality and asset fidelity: the experience contains no source logo, illustration, or product-image asset that needs reproduction. The graph is rendered by Cytoscape.js canvas and remains crisp; no placeholder imagery is present.
- Copy and content: labels are grounded in `docs/multi_agent_driving_mvp_spec.md`. Each of the four Flow cards now includes a compact description inside its border, and each Flow exposes actor, trigger, outcome, stages, backstage work, outputs, safety/recovery checks, specification evidence, and a coverage gap. The `Spec-backed` badge and coverage notes accurately distinguish specification evidence from unverified runtime behavior.
- Interaction and accessibility: exact-label search, direct node clicking, four-card Flow navigation, stage-to-node navigation, direct-relationship navigation, and fit-to-view all passed in the browser. Selecting a Flow updates both the focused graph and the service blueprint, sets `aria-pressed`, updates the shareable hash, and automatically scrolls to the chosen detail. At `1280×720`, `Training loop` selection scrolled the inspector to `scrollTop=324`; scrolling upward returned to `scrollTop=0`, where all four cards were visible. At `390×844`, selecting `Evidence & evaluation` placed its heading 56px below the viewport top; scrolling upward exposed the full four-card selector from 26px to 362px. The former top Flow selector is absent from both the DOM and rendered toolbar. Native controls expose accessible names and pressed states. Browser console check returned no warnings or errors.

## Comparison History

### Pass 1

- [P2] The desktop toolbar wrapped to multiple rows at the reference viewport, consuming too much vertical space.
- [P2] The initial graph was too small and low-density because the full 2,210-unit coordinate range was fitted at once and only one agent branch was emphasized.
- Fixes: constrained picker widths, hid nonessential header metadata below 1,400px, removed deprecated/custom Cytoscape settings, highlighted all branches of the decision cycle, increased the focused-flow zoom, and added responsive flow fitting.
- Post-fix evidence: `docs/qa/design-qa-comparison-pass1.png`.

### Pass 2

- [P2] The graph still sat lower than the source, leaving an oversized gap under the title.
- Fix: shifted the focused desktop graph upward while preserving the mobile fit and inspector layout.
- Post-fix evidence: `docs/qa/design-qa-comparison-pass2.png`.

### Pass 3 — Flow inspector

- [P1] Selecting a Flow focused the graph, but the right inspector still described the previously selected node, so the chosen scenario was not presented as a complete Flow.
- [P2] Adding stage-rich Flow content initially expanded the desktop workspace height and centered the canvas below the viewport.
- Fixes: added a Flow/service-blueprint inspector, cleared stale node selection on Flow change, added shareable `#flow=` state, separated desktop viewport height from the inspector scroll region, and refit the active Flow below the title area.
- Post-fix evidence: `docs/qa/design-qa-comparison-pass3.png` and `docs/qa/implementation-flow-final.png`.

### Pass 4 — Persistent Flow navigation

- [P1] Flow choices were replaced by the selected Flow detail, making it unnecessarily difficult to return to the full architecture after reading a long inspector.
- [P2] The top node picker duplicated direct node clicking, while the node-color legend sat at the bottom of the graph and was easy to miss.
- Fixes: removed the node picker, moved the color legend under the map description, placed all four Flow views above the right-side content, kept the selector sticky on desktop, and added an automatic mobile scroll to the selected Flow heading with the choices preserved immediately above it.
- Post-fix evidence: `docs/qa/design-qa-comparison-pass4.png`, `docs/qa/implementation-flow-navigation-final.png`, and `docs/qa/implementation-flow-mobile-auto-scroll.png`.

### Pass 5 — Single large Flow-card navigation

- [P2] The toolbar still contained a second Flow selector after the right-side choice set became the primary navigation.
- [P2] The 2×2 Flow buttons were too small to explain what each view contains and did not match the requested 4-row reading order.
- Fixes: removed the top Flow label/select and its JavaScript synchronization path; changed the right-side selector to four large, description-bearing cards in one column; moved the selector into normal scroll flow; retained automatic scrolling to selected content and simple upward scrolling back to the choices.
- Post-fix evidence: `docs/qa/design-qa-comparison-pass5.png`, `docs/qa/implementation-flow-cards-final.png`, `docs/qa/implementation-flow-cards-mobile-options.png`, and `docs/qa/implementation-flow-cards-mobile.png`.

## Open Questions

- None blocking. The implementation intentionally adapts the reference into a Japanese, repository-specific exploration tool rather than cloning the source video's exact product chrome.

## Implementation Checklist

- [x] Match the dark technical map composition and amber primary flow.
- [x] Keep the graph and right-side inspector functional.
- [x] Expose machine-readable architecture context in a separate JSON file.
- [x] Mark specification-only modules as planned.
- [x] Verify desktop and mobile layout.
- [x] Verify search, direct node clicking, four-card Flow switching, return-to-choice navigation, relationship navigation, fit control, and console state.
- [x] Verify Flow selection changes graph focus and the right-side Flow/service-blueprint content.
- [x] Verify the top Flow selector is removed.
- [x] Verify four large description-bearing choices render in one column on desktop and mobile.
- [x] Verify automatic detail scrolling and upward return to the choices on desktop and mobile.

## Follow-up Polish

- [P3] If the module count grows substantially, hide edge labels below a zoom threshold to keep the mobile canvas quieter.

final result: passed
