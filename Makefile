
# ----------------------------------
#         Vincent
# ----------------------------------

PROJECT_ID=possible-flag-305109
DOCKER_IMAGE_NAME=apioff

run_local:
	-@ uvicorn api.fast:app --reload

docker_build:
	-@ docker build -t eu.gcr.io/${PROJECT_ID}/${DOCKER_IMAGE_NAME} .

docker_run:
	-@ docker run -e PORT=8000 -p 8000:8000 eu.gcr.io/${PROJECT_ID}/${DOCKER_IMAGE_NAME}

cloud_set_project:
	-@gcloud config set project ${PROJECT_ID}

cloud_push:
	-@ gcloud

cloud_deploy:
	-@ gcloud

# ----------------------------------
#          INSTALL & TEST
# ----------------------------------
install_requirements:
	@pip install -r requirements.txt

check_code:
	@flake8 scripts/* OpenFoodFactsApi/*.py

black:
	@black scripts/* OpenFoodFactsApi/*.py

test:
	@coverage run -m pytest tests/*.py
	@coverage report -m --omit="${VIRTUAL_ENV}/lib/python*"

ftest:
	@Write me

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr */__pycache__ */*.pyc __pycache__
	@rm -fr build dist
	@rm -fr OpenFoodFactsApi-*.dist-info
	@rm -fr OpenFoodFactsApi.egg-info

install:
	@pip install . -U

all: clean install test black check_code


uninstal:
	@python setup.py install --record files.txt
	@cat files.txt | xargs rm -rf
	@rm -f files.txt

count_lines:
	@find ./ -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./scripts -name '*-*' -exec  wc -l {} \; | sort -n| awk \
		        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./tests -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''

# ----------------------------------
#      UPLOAD PACKAGE TO PYPI
# ----------------------------------
PYPI_USERNAME=<AUTHOR>
build:
	@python setup.py sdist bdist_wheel

pypi_test:
	@twine upload -r testpypi dist/* -u $(PYPI_USERNAME)

pypi:
	@twine upload dist/* -u $(PYPI_USERNAME)
