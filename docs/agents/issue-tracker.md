# Issue tracker: GitHub

Issues and specs live in AmanurRahman101/AUTODSPy.
Use the gh CLI from this clone.

## Operations

- Create: gh issue create --title "..." --body-file <file>
- Read: gh issue view <number> --json number,title,body,labels,comments
- List: gh issue list --state open --json number,title,body,labels
- Comment: gh issue comment <number> --body-file <file>
- Add labels: gh issue edit <number> --add-label "..."
- Remove labels: gh issue edit <number> --remove-label "..."
- Close: gh issue close <number> --comment "..."

Write multiline bodies to a UTF-8 file and pass --body-file.
Use --repo AmanurRahman101/AUTODSPy when outside this clone.

## Pull requests as a triage surface

PRs as a request surface: no.

## Skill instructions

"Publish to the issue tracker" means create a GitHub issue.
"Fetch the relevant ticket" means read the referenced GitHub issue.
