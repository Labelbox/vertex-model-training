import setuptools

setuptools.setup(
    name="vertex",
    version="0.0.1",
    author="Labelbox",
    author_email="engineering@labelbox.com",
    description="shared code for model training jobs and the coordinator service",
    packages=setuptools.find_packages(),
    install_requires=[
        "PILLOW", "google-cloud-storage",
        "labelbox[data]", "requests", "google-api-core"],
)
