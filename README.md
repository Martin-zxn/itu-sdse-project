# ITU BDS MLOPS'26 - Project (V2)

## Task

Based on the input provided (see below), fork the repository and restructure the code to adhere to the concepts and ideas you have seen throughout the course. The diagram below provides a detailed overview of the structure that the solution is expected to follow.

![Project architecture](./docs/project-architecture.png)

For the exam submission, we expect you to submit a pdf containing:
- the list of members of the group
- the link to the github.com public repository hosting your solution
  - following the above, there is *no need* to invite the teaching staff as collaborators

The repository linked in the submission should contain:

- A README.md file that describes the project
- GitHub automation workflow
- Dagger workflow (in Go)
- All history

## Inputs

You are given the following material:
- Python monoliths (see `notebooks` folder)
- Raw input data (pulled from the object store by the training notebook)
- Group login to https://sdse-event-engine-r5ih8.ondigitalocean.app/ for monitoring deployed model
- DigitalOcean App Platform spec for the deployment (see `infra/digitalocean/app.yaml`)

## Outputs

- Your GitHub repository (including all history)
  - A README.md file that describes the project
  - GitHub automation workflow
  - Dagger workflow (in Go)
- Model artifacts produced by GitHub workflow
- Hosted endpoint in DigitalOcean
  - /health that returns {"status": "ok"}
  - /invocations that take binary image payload
  - /drift that returns model/data drift information

> **NOTE:**
> The Dagger workflow can be run locally or inside the GitHub workflow—both are viable options during development.
>
> The Dagger workflow can run locally and can also be made to produce outputs locally during development. But when wrapping the Dagger workflow in a GitHub workflow, the output is instead stored inside the GitHub runner (i.e. a virtual machine).
>
> Use the publicly available [`actions/upload-artifact`](https://github.com/actions/upload-artifact) to store the model artifact in the GitHub worklow pipeline.
>
> This model artifact can then be picked up by the [action provided](https://github.com/lasselundstenjensen/itu-sdse-project-model-validator), which will run some inference tests to ensure that the correct model was trained.

### Refactored CNN training

After pulling the DVC metadata and downloading the matching image files into
`data/images/raw/`, train the notebook-equivalent CNN from the repository root:

```bash
python -m modeling.train
```

The command reads `data/images/metadata.csv`, trains the `SimpleCNN` with the
notebook's 64x64 preprocessing, 70/15/15 stratified split, and training
settings, then writes `artifacts/vial_cnn_model.pth`,
`artifacts/model_results.json`, `artifacts/class_names.json`,
`artifacts/image_statistics.json`, and
`artifacts/drift_baseline_stats.json`.
The checkpoint contains the notebook-compatible `model_state_dict`,
`architecture`, `input_shape`, `output_shape`, `threshold`, and `num_classes`
fields. Override data and checkpoint paths with `--metadata`, `--images`, and
`--output`; use `python -m modeling.train --help` to see all options.
Run the model and drift checks with `python -m unittest discover -s tests`.

The linked validator action currently expects a scikit-learn `model.pkl` and
tabular predictions, which is a different artifact contract from this
notebook's PyTorch image classifier checkpoint. The `.pth` checkpoint retains
the image-model format expected by the notebook's model loader.

## How will we assess

Below, we provide information on how we will assess the submission clustered around several aspects. The groups are expected to be of size 4, and you're expected to use pull requests and tests.

### Versioning

- Use of Git (semantic commit messages, branches, branch longevity, commit frequency/size)
- Management of data and artifacts
- Use of pull requests

### Programming

- Decomposition of Python notebook
- Adherance to standard data science MLOps project structure
- Presence of tests

### Workflow automation

- Presence of a workflow that trains the model
- Presence of a workflow that tests the model
- Structure of Dagger workflow
- Orchestration of Dagger workflow through GitHub workflow

### Documentation (README.md)

- Description of project structure
- How to run the code and generate the model artifact

## Questions

If you have any questions about the information shared here, please feel free to post them on Learnit. Answers to private emails on this topic will also be shared on Learnit, along with the original email content, so that everyone has access to the same information.
