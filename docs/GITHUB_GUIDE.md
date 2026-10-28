# Publish the project to GitHub

1. Sign in to GitHub and create a **public** repository named `COMP8240-OptIForest-Replication`. Do not initialise it with a README if you already have this local project README.
2. Unzip this package and open Terminal in the extracted `COMP8240_OptIForest_Final_Report_Pack` folder.
3. Review files before adding them. Do not commit private credentials, the user's private data, `.venv/`, caches, or a copied upstream OptIForest repository.
4. Run:

```bash
git init
git branch -M main
git add README.md .gitignore LICENSE requirements.txt requirements-optiforest-legacy.txt src tests data figures results report docs scripts prompts_llm_judge.md
git status
git commit -m "Add reproducible OptIForest replication project"
git remote add origin https://github.com/<USERNAME>/COMP8240-OptIForest-Replication.git
git push -u origin main
```

5. Replace `<USERNAME>` with your GitHub username.
6. Open the repository in a private/incognito browser window to confirm it is public.
7. Keep the upstream source at <https://github.com/xiagll/OptIForest> as a link. Its licence was not visible in the repository view inspected for this project, so this project does not redistribute the upstream implementation.
8. After logging official OptIForest runs, commit the metrics tables, raw output and run metadata needed for the marker to reproduce the report. Avoid committing huge datasets if their source, version/checksum and generation script are enough—but the project’s synthetic dataset is small enough to include.

The final report must include the URL of this public repository in the replication section, as specified by the assignment brief.
