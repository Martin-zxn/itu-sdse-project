package main

import (
	"context"
	"fmt"
	"os"

	"dagger.io/dagger"
)

func main() {
	ctx := context.Background()

	client, err := dagger.Connect(ctx, dagger.WithLogOutput(os.Stderr))
	if err != nil {
		fmt.Fprintln(os.Stderr, "could not connect to Dagger:", err)
		os.Exit(1)
	}
	defer client.Close()

	// The project folder (one level above dagger/). Data, caches and local
	// build folders are left out, so the container starts from a clean state.
	src := client.Host().Directory("..", dagger.HostDirectoryOpts{
		Exclude: []string{
			"dagger", "artifacts", ".venv", "**/__pycache__",
			".dvc/cache", "data/images/metadata.csv", "data/images/raw",
			"data/test/metadata.csv",
		},
	})

	python := client.Container().
		From("python:3.13-slim").
		WithEnvVariable("PYTHONPATH", "src").
		WithEnvVariable("DVC_NO_ANALYTICS", "1").
		WithExec([]string{"pip", "install", "--no-cache-dir", "boto3", "dvc"}).
		WithDirectory("/app", src).
		WithWorkdir("/app")

	// Step 1: get data (real)
	// DVC cannot restore the metadata in a fresh container (no cache, no remote),
	// so the two metadata CSVs are downloaded straight from the object store.
	fetchMetadata := `
import pathlib, urllib.request
base = "https://itudsip.hel1.your-objectstorage.com/data/"
for folder in ("images", "test"):
    path = pathlib.Path("data") / folder / "metadata.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(base + folder + "/metadata.csv", path)
`
	out, err := python.
		WithExec([]string{"python", "-c", fetchMetadata}).
		WithExec([]string{"sh", "-c", "python -c 'from ccds.dataset import pull_data_from_dvc; pull_data_from_dvc()' && ls data/images/raw | wc -l"}).
		Stdout(ctx)
	if err != nil {
		fmt.Fprintln(os.Stderr, "step failed: get data", err)
		os.Exit(1)
	}
	fmt.Print(out)

	// Steps 2 to 4: still placeholders
	for _, step := range []string{"prepare data", "train model", "save model"} {
		out, err := runStep(ctx, client, step)
		if err != nil {
			fmt.Fprintln(os.Stderr, "step failed:", step, err)
			os.Exit(1)
		}
		fmt.Print(out)
	}
}

// runStep is a placeholder. Later it will run the real commands.
func runStep(ctx context.Context, client *dagger.Client, name string) (string, error) {
	return client.Container().
		From("python:3.13-slim").
		WithExec([]string{"echo", "PLACEHOLDER: " + name}).
		Stdout(ctx)
}
