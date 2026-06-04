from setuptools import find_packages, setup

package_name = "my_vision_policy"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    package_data={package_name: ["weights/*.pt"]},
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="gr",
    maintainer_email="gruzver.phocco@pucp.edu.pe",
    description="Vision-based cable insertion policy",
    license="Apache-2.0",
)
