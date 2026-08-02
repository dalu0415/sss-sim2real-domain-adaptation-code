# Data metadata and access boundary

The study used a 490-image semi-synthetic source domain. No source PNG files,
label tables, release archive, or image-data licence are distributed in this
repository.

`manifest.csv` is retained as research metadata. It records the relative
filenames, class labels, file types, byte sizes, SHA-256 digests, dimensions,
and image modes of the source-domain files used in the manuscript. It does not
provide the image assets, constitute a download location, or grant a licence
to any image, input silhouette, or upstream material.

The study also used the third-party SeabedObjects-KLSG-II target dataset. Its
authors' distribution repository is listed in the manuscript and root README;
this repository does not include or redistribute its image bytes.

The code may be used with data that a user lawfully obtains or constructs under
the relevant terms. Required directory layouts and known non-self-contained
simulation inputs are documented in the root README and
[`src/simulation/README.md`](../src/simulation/README.md).
