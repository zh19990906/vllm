from pathlib import Path
import json

DELIVERY = "2752b4950f0f30eedbb7f6bb3b60a83512a012c4"
RUN_ID = 31682582711

p = Path("docs/engineering/validation/2026-08-12-issue31-filesystem-hard-capacity-validation.json")
data = json.loads(p.read_text())
data["ci"] = {
    "authoritative": True,
    "status": "implementation_head_passed_docs_refresh_pending",
    "workflow": "pre-commit",
    "run_id": RUN_ID,
    "run_number": 202,
    "latest_attempt_conclusion": "success",
    "note": (
        "authoritative pre-commit run #202 passed on implementation head "
        f"{DELIVERY}; this documentation-only refresh requires a fresh CI run "
        "after publication before merge"
    ),
}
data["github"].update({
    "draft_pr_status": "open",
    "draft_pr_number": 33,
    "implementation_branch_head": DELIVERY,
    "local_implementation_not_yet_published": False,
})
data["delivery_head_sha"] = DELIVERY
data["status"] = "implementation_ci_passed_docs_refresh_pending"
data["focused_verification"]["pytest"] = {
    "available_in_pod": False,
    "status": "not_run_no_pytest_ci_job_observed_for_pr33",
}
p.write_text(json.dumps(data, indent=2) + "\n")

p = Path("docs/engineering/validation/2026-08-12-issue31-filesystem-hard-capacity-validation.md")
text = p.read_text()

def ro(old, new):
    global text
    if text.count(old) != 1:
        raise SystemExit(f"validation replace mismatch: {old[:80]!r}")
    text = text.replace(old, new, 1)

ro(
    "Issue #31 is **locally validated with authoritative GitHub CI still pending**.",
    "Issue #31 is **locally validated and authoritative GitHub pre-commit CI passed on implementation head `2752b4950f0f30eedbb7f6bb3b60a83512a012c4`; this documentation-only refresh requires a fresh CI run after publication before merge**.",
)
ro(
    """The implementation has not yet been published from the Pod-local implementation head
to the GitHub Issue #31 branch, no Draft PR exists yet, and authoritative GitHub CI
has not run.""",
    """The implementation is published as Draft PR #33 at
`agent/issue31-fs-hard-capacity@2752b4950f0f30eedbb7f6bb3b60a83512a012c4`.
Authoritative GitHub pre-commit run #202 (`31682582711`) passed on its latest
attempt. The PR remains Draft and has not been merged.""",
)
ro(
    """- GitHub branch `agent/issue31-fs-hard-capacity` still pointed to
  `eea0ff4b16711693b6f9945a4a808916990442ee` before publication.""",
    """- GitHub branch `agent/issue31-fs-hard-capacity` now points to
  `2752b4950f0f30eedbb7f6bb3b60a83512a012c4`.
- Draft PR #33 is open and remains Draft.
- Authoritative GitHub pre-commit run #202 (`31682582711`) passed on its latest
  attempt.""",
)
ro(
    """- Pytest is unavailable in the Pod; pytest-only coverage remains an authoritative
  GitHub-CI gate.""",
    """- Pytest is unavailable in the Pod; no pytest CI job was observed for PR #33
  at the current delivery head. Focused Issue #31 unittest coverage is recorded
  below.""",
)
ro("| Pod pytest | unavailable; deferred to GitHub CI |", "| Pod pytest | unavailable; no pytest CI job observed on PR #33 |")
ro(
    """covered locally plus the existing inference architecture; authoritative repository CI is
still required before merge.""",
    """covered locally plus the existing inference architecture. Authoritative GitHub
pre-commit CI has passed; final review and explicit user merge authorization are
still required.""",
)
ro(
    """**Issue #16 remains blocked until Issue #31 is published, passes authoritative GitHub CI,
is merged, and is closed.**""",
    """**Issue #16 remains blocked until Issue #31 is merged and closed. Publication and
authoritative GitHub pre-commit CI are complete.**""",
)
marker = "## CI and merge status\n"
prefix = text.split(marker, 1)[0]
text = prefix + """## CI and merge status

Local/delivery status: `implementation_ci_passed_docs_refresh_pending`.

Draft PR #33 is open at delivery head
`2752b4950f0f30eedbb7f6bb3b60a83512a012c4`.

Authoritative GitHub pre-commit run #202 (`31682582711`) passed on its latest
attempt on implementation head
`2752b4950f0f30eedbb7f6bb3b60a83512a012c4`. An earlier attempt exposed
repository-wide ShellCheck findings in files outside the Issue #31 diff;
rerunning the same head passed. The locally investigated ShellCheck wrapper
change is not part of Issue #31.

This documentation-only refresh will create a newer delivery head. That new
head must receive a fresh authoritative CI result before merge.

Pytest is unavailable in the Pod, and no pytest CI job was observed for the
current PR head. The recorded focused unittest, smoke-contract, compile, mypy,
repository-policy, and filesystem-smoke evidence therefore remains the relevant
Issue #31 validation evidence.

The PR remains Draft. Green CI is not merge authorization. The next delivery
stage is final review followed by explicit user authorization before any merge.
Issue #16 remains blocked until Issue #31 is merged and closed.
"""
p.write_text(text)

p = Path("docs/engineering/CURRENT_STATE.md")
text = p.read_text()

def rs(old, new):
    global text
    if text.count(old) != 1:
        raise SystemExit(f"state replace mismatch: {old[:80]!r}")
    text = text.replace(old, new, 1)

rs(
    """The implementation is locally validated at
`949beed012b57281ae8eadd63cc8a674fb1975e0`, with formal filesystem evidence in:""",
    """The formal filesystem validation was produced from implementation head
`949beed012b57281ae8eadd63cc8a674fb1975e0`. The current delivery head is
`2752b4950f0f30eedbb7f6bb3b60a83512a012c4`, which retains that behavior plus
repository quality-gate remediation. Formal filesystem evidence is recorded in:""",
)
rs(
    """The GitHub implementation branch has not yet been updated from its plan-only head and
no Draft PR/authoritative CI result exists yet. Do not describe Issue #31 as merged or
closed.""",
    """GitHub branch `agent/issue31-fs-hard-capacity` now points to
`2752b4950f0f30eedbb7f6bb3b60a83512a012c4`. Draft PR #33 is open, and
authoritative GitHub pre-commit run #202 passed on its latest attempt. Do not
describe Issue #31 as merged or closed.""",
)
rs(
    """Issue #31 has completed local implementation and local filesystem validation. Remaining
work is repository delivery and authoritative CI, not additional local feature expansion.""",
    """Issue #31 has completed local implementation, local filesystem validation, repository
publication, and authoritative GitHub pre-commit CI. Remaining work is final review
and explicit merge authorization, not additional local feature expansion.""",
)
start = "## Current continuation record\n"
end = "## Important interpretation rules\n"
before, rest = text.split(start, 1)
_, after = rest.split(end, 1)
continuation = """## Current continuation record

Live GitHub `main` was observed at
`c4d9fce61ec5a8eadc24dab8698eca7705d005bf`.

Draft PR #33 targets `main` from
`agent/issue31-fs-hard-capacity@2752b4950f0f30eedbb7f6bb3b60a83512a012c4`.

Authoritative GitHub pre-commit run #202 (`31682582711`) passed on its latest
attempt on implementation head
`2752b4950f0f30eedbb7f6bb3b60a83512a012c4`. This documentation-only refresh
will create a newer delivery head, which requires a fresh authoritative CI run
after publication. The PR remains Draft and unmerged. Green CI is not merge
authorization.

Pytest is unavailable in the Pod, and no pytest CI job was observed for the
current PR head. Focused Issue #31 unittest, smoke-contract, mypy, repository
policy, compile, and formal real-filesystem evidence remain recorded in the
validation artifact.

The next step is final review. Merge requires explicit user authorization.
Issue #16 remains blocked until Issue #31 is merged and closed.

"""
p.write_text(before + continuation + end + after)
