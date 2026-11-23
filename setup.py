# Copyright 2023-2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).resolve().parent
requirements = [line.strip() for line in (ROOT / "requirements.txt").read_text().splitlines()
                if line.strip() and not line.lstrip().startswith("#")]

setup(
    name="verl",
    version=(ROOT / "verl/version/version").read_text().strip(),
    packages=find_packages(include=["verl", "verl.*", "corpus", "evaluation", "training"]),
    py_modules=["conversation"],
    python_requires=">=3.11",
    url="https://github.com/haikunhere/CoSTA",
    license="Apache-2.0",
    license_files=["LICENSE", "verl/LICENSE"],
    description="CoSTA training, inference, and evaluation",
    install_requires=requirements,
    package_data={
        "verl": [
            "LICENSE",
            "py.typed",
            "version/*",
            "trainer/config/*.yaml",
            "trainer/config/*/*.yaml",
            "experimental/*/config/*.yaml",
        ],
    },
    include_package_data=True,
    long_description=(ROOT / "README.md").read_text(),
    long_description_content_type="text/markdown",
)
