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

	// Exclude local data and build folders but keep the DVC cache.
	src := client.Host().Directory("..", dagger.HostDirectoryOpts{
		Exclude: []string{
			"dagger", "artifacts", ".venv", "**/__pycache__",
			"data/images/metadata.csv", "data/images/raw",
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

	// Step 1: restore metadata from DVC and download images
	out, err := python.
		WithExec([]string{"dvc", "checkout"}).
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
