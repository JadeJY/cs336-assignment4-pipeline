# CS336 Spring 2025 Assignment 4: Data

For a full description of the assignment, see the assignment handout at
[cs336_spring2025_assignment4_data.pdf](./cs336_spring2025_assignment4_data.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

This directory is organized as follows:

- [`./cs336-basics`](./cs336-basics): Ass1 model 的标准实现
- [`./cs336_data`](./cs336_data): 包含所有的数据处理文件

Visually, it should look something like:

``` sh
.
├── cs336_basics  # A python module named cs336_basics
│   └── ... an optimized training implementation ...
├── cs336_data  # TODO(you): code that you'll write for assignment 4
│   └── __init__.py
│   └── deduplication.py
│   └── pipeline.py
│   └── preprocessing.py
│   └── preprocessing.py
├── README.md
├── pyproject.toml
└── scripts
    └── scrape_positive.sh
    
```

As in previous assignments, we use `uv` to manage dependencies.

## Submitting

To submit, run `./test_and_make_submission.sh` . This script will install your
code's dependencies, run tests, and create a gzipped tarball with the output. We
should be able to unzip your submitted tarball and run
`./test_and_make_submission.sh` to verify your test results.
