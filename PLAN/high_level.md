# OVERWATCH — High-Level Plan

A non-technical companion to `low_level.md`. Same project, written for a supervisor, partner, or product owner.

---

## What this is

Security cameras record around the clock. A mid-sized facility with 40 cameras produces roughly 960 hours of footage every day. When something goes wrong, investigators scrub through timeline bars hunting for the right three seconds. We let them type a description instead. "A man getting into a white van" returns the matching frame in seconds, with the subject highlighted in a box. It is forensic search for the archive every camera already produces.

---

## Where it stands today

We have a working prototype, deployed on the public web, that an investigator can use end-to-end: upload footage, type a query in plain English, and get back the matching moments ranked by relevance. The system understands compositional descriptions ("a child in a red shirt and black cap") and verifies its top picks with a second AI layer before showing them to the user, which keeps obvious false positives off the screen.

This is a working forensic-retrieval prototype, not a finished surveillance product. It returns answers in seconds on the demo dataset. It is not yet hardened for paying customers, multiple tenants, or always-on operation.

---

## What it is good for

- **Insurance investigations** — vehicle theft, property damage, slip-and-fall claims.
- **Unauthorised access** — server rooms, restricted zones, after-hours entries.
- **Emergency-response audits** — confirming when ambulances or responders actually arrived.
- **Missing-person searches** — describing a child or vulnerable adult in plain language and pulling matching frames from public-area cameras.
- **Retail loss prevention** — finding the moment of a shoplifting incident and identifying repeat offenders.
- **Litigation discovery** — searching long blocks of footage for events that match a legal description.
- **HR and harassment investigations** — locating specific interactions in office CCTV.

The common thread: someone already knows roughly what happened, and needs the visual evidence quickly.

---

## What it isn't

- **Not real-time monitoring.** It reacts to a query; it does not watch live feeds 24/7 and raise the alarm.
- **Not an alerting system.** There are no subscriptions or push notifications yet.
- **Not a video-analytics suite.** It does not count people, draw heatmaps, or measure dwell time.
- **Not multi-camera live coordination.** It works on one piece of footage at a time today.

These are roadmap items, not architectural blockers, but they are not in the box on day one.

---

## Risks to address before customers see it

These are the issues a buyer or auditor would raise. Listed in the order we would tackle them.

1. **Data integrity in the ingestion pipeline.** One ingestion path can write incomplete records that then pollute every search result. A workaround is in place; the underlying fix is mandatory before customer onboarding. Severity: high. Mitigation: rebuild the pipeline so no record enters the index unverified.

2. **No real authentication.** Administrative actions — including wiping the dataset — are protected only by a placeholder password. Severity: high. Mitigation: a proper login layer and per-account permissions. Days, not weeks.

3. **Single-tenant only.** Every customer would see the same dataset. Severity: medium today, high the moment a second customer signs. Mitigation: account-scoped indexes.

4. **No automated quality gate.** Every change is tested by hand; regressions will slip through silently as the team grows. Severity: medium. Mitigation: a baseline test suite and a deploy-on-green pipeline.

5. **Capacity ceiling.** The current hosting tier handles a small number of simultaneous users and pauses for over a minute when idle. Fine for demos, not for launch. Severity: medium. Mitigation: a modest hosting upgrade — better hardware that runs the AI models faster and stays warm — in the low hundreds of dollars per month.

6. **Operational visibility.** When something breaks in production, we find out because a user tells us. Severity: medium. Mitigation: standard error-tracking and uptime monitoring tools.

None of these are research problems. They are engineering hygiene.

---

## Where it goes next

Three product milestones, ordered by customer impact.

**Milestone 1 — Production-ready forensic search.** Close the data-integrity, authentication, and tenancy gaps above; move to the upgraded hosting tier; add the basic operational tooling. The product becomes safe to put in front of a paying pilot customer for the use cases it already serves.

**Milestone 2 — From search tool to surveillance product.** Add saved searches, alerts when new footage matches a saved description, an incident-timeline view that shows every match from a single video on one screen, and an evidence-export feature that produces a downloadable package suitable for insurance or legal handoff. This is the step that takes us from "useful utility" to "product a security team builds their workflow around."

**Milestone 3 — Becoming infrastructure other AI tools call into.** Today the only way to use the system is through our web interface. The strategic move is to expose the same search capability as a service that other AI assistants — the chatbots and copilots investigators are starting to use day-to-day — can call directly. (The emerging standard for this is called Model Context Protocol, MCP for short; it lets AI assistants plug into external tools uniformly.) An investigator could then ask their assistant in plain language to find a moment in yesterday's footage, and the assistant would query our system behind the scenes and show the answer inline. The user interface is one consumption surface; becoming the back end behind every AI-native security tool is a far stronger commercial position than competing with the next prettier UI.

---

## What we need from you

Four decisions are blocked on stakeholder input, not engineering.

- **Ownership and handoff.** All cloud accounts, model hosting, and data repositories sit under one contributor's personal accounts. Before any commercial step, these need to move to organisational ownership. Who takes them on, and when?
- **Hosting budget.** A modest hardware upgrade — from the free tier to a small dedicated AI-capable server — would unblock the capacity, cold-start, and tracking-quality issues at once. A few hundred dollars per month. Approved?
- **Positioning.** Do we lead with forensic, post-incident search (today's strength), or invest in live monitoring (a different, more expensive architecture)? Both can coexist; we need to know which one comes first.
- **Licensing and handoff terms.** If WyseTime Technologies wants to take this forward, is it a licensed product, a research prototype handed over outright, or a continuing collaboration? Each implies different IP and engineering arrangements.

Clear answers unlock the production-readiness milestone immediately.
