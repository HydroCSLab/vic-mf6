# Releasing the bundle

A bundle release should identify source that another person can fetch and an
image that has already passed the complete acceptance workflow. Prepare it in
this order:

1. Push the reviewed VIC and MODFLOW 6 branches to their public forks. Confirm
   that each commit in `components.lock` is reachable from the repository
   recorded on the same line. The coupler commit is the parent `vic-mf6`
   commit being released.
2. Review and merge the component pull requests according to each upstream
   project. Component history remains in the component repository; do not copy
   patches into the bundle.
3. Update each submodule to the final reviewed commit. Point a component at its
   authoritative upstream repository once that commit is available there, then
   update `components.lock`.
4. Build the image from a clean recursive clone and run
   `./bundle/scripts/run-acceptance.sh`. Review the numerical acceptance report and
   `software-environment.txt` before assigning a version tag.
5. Publish the tested image and record its immutable registry digest in the
   release notes. Keep the human-readable version tag for convenience, but use
   the digest in manuscript and archival reproducibility records.

## GitHub Container Registry

The bundle image is intended for GitHub Container Registry as
`ghcr.io/mabdazzam/vic-mf6`.
Build and test a versioned image before authenticating or publishing it:

```bash
export VICMF6_VERSION=0.1.0
./bundle/scripts/build-image.sh ghcr.io/mabdazzam/vic-mf6:$VICMF6_VERSION
VICMF6_IMAGE=ghcr.io/mabdazzam/vic-mf6:$VICMF6_VERSION \
    ./bundle/scripts/run-verification-evidence.sh
VICMF6_IMAGE=ghcr.io/mabdazzam/vic-mf6:$VICMF6_VERSION \
    ./bundle/scripts/run-acceptance.sh bundle/results/release-stehekin
```

After reviewing the exact commit, tests, and image tag, authenticate with a
GitHub personal access token that has the `write:packages` scope and publish:

```bash
printf '%s' "$CR_PAT" | docker login ghcr.io --username mabdazzam --password-stdin
docker push ghcr.io/mabdazzam/vic-mf6:$VICMF6_VERSION
```

Set the package visibility deliberately in GitHub after the first push.
Record the resulting immutable digest from `docker inspect` in the release
notes and use the digest in archival instructions.
The Dockerfile records OCI source, version, and revision labels; override
`VICMF6_SOURCE_REPOSITORY` only if the public bundle repository has a different
canonical URL.

Pushing branches, opening pull requests, merging, tagging, and publishing an
image are external actions. They should be performed only after the exact
local commits, diff, test record, and target repository have been reviewed.
