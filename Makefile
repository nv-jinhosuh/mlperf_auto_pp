SHELL := /bin/bash
MAKEFILE_NAME := Makefile 
ARCH := $(shell uname -p)
UNAME := $(shell whoami)
UID := $(shell id -u `whoami`)
HOSTNAME := $(shell hostname)
GROUPNAME := $(shell id -gn `whoami`)
GROUPID := $(shell id -g `whoami`)

NVIDIA_VISIBLE_DEVICES ?= all
NVIDIA_DRIVER_CAPABILITIES ?=compute,utility

NGC_VER = 24.03
PY_VER = 310

TRT_DEB_URL = http://cuda-repo.nvidia.com/release-candidates/Libraries/TensorRT/v9.3/9.3.0.1-d6cbd29d/12.2-r535/Ubuntu22_04-${ARCH}/deb/
TRT_MAJOR_VER=9
TRT_MINOR_VER=3
TRT_PATCH_VER=0
TRT_QA_VER=1
CUDA_VER = 12.2

DOCKER_BASE_IMAGE ?= nvcr.io/nvidia/pytorch:$(NGC_VER)-py3
DOCKER_IMAGE_NAME ?= mlperf-auto
DOCKER_TAG := $(UNAME)
DOCKER_NAME := $(DOCKER_IMAGE_NAME)--$(DOCKER_TAG)

DOCKER_FILENAME ?= docker/Dockerfile.$(ARCH).build
DOCKER_WAYMO_FILENAME ?= docker/Dockerfile.$(ARCH).waymo

ifeq ($(DOCKER_COMMAND),)
	DOCKER_INTERACTIVE_FLAGS = -it
else
	DOCKER_INTERACTIVE_FLAGS =
endif

HOST_VOL ?= ${PWD}
CONTAINER_VOL ?= /work

ifneq ($(wildcard /home/scratch.svc_compute_arch),)
    DOCKER_MOUNTS += -v /home/scratch.svc_compute_arch:/home/scratch.svc_compute_arch
endif
ifneq ($(wildcard /home/scratch.computelab/sudo),)
    DOCKER_MOUNTS += -v /home/scratch.computelab/sudo:/home/scratch.computelab/sudo
endif

# This is where the waymo open dataset is
ifneq ($(wildcard /home/scratch.jsuh_gpu_5),)
	DOCKER_MOUNTS += -v /home/scratch.jsuh_gpu_5:/home/scratch.jsuh_gpu_5
endif

DOCKER_RUN_CMD := nvidia-docker run

.PHONY: build_docker
build_docker:
	docker build -t $(DOCKER_IMAGE_NAME):$(DOCKER_TAG)-latest \
		--build-arg DOCKER_BASE_IMAGE=$(DOCKER_BASE_IMAGE) \
		--build-arg TRT_DEB_URL=$(TRT_DEB_URL) \
		--build-arg TRT_MAJOR_VER=$(TRT_MAJOR_VER) \
		--build-arg TRT_MINOR_VER=$(TRT_MINOR_VER) \
		--build-arg TRT_PATCH_VER=$(TRT_PATCH_VER) \
		--build-arg TRT_QA_VER=$(TRT_QA_VER) \
		--build-arg CUDA_VER=$(CUDA_VER) \
		--network host \
		-f $(DOCKER_FILENAME) \
		.

.PHONY: build_waymo_docker
build_waymo_docker:
	docker build -t $(DOCKER_IMAGE_NAME):$(DOCKER_TAG)-latest \
		--build-arg DOCKER_BASE_IMAGE=$(DOCKER_BASE_IMAGE):$(DOCKER_TAG)-latest \
		--network host \
		-f $(DOCKER_WAYMO_FILENAME) \
		.

.PHONY: docker_add_user
docker_add_user:
	docker build -t $(DOCKER_IMAGE_NAME):$(DOCKER_TAG) \
		--build-arg DOCKER_BASE_IMAGE=$(DOCKER_IMAGE_NAME):$(DOCKER_TAG)-latest \
		--build-arg GID=$(GROUPID) \
		--build-arg UID=$(UID) \
		--build-arg GROUP=$(GROUPNAME) \
		--build-arg USER=$(UNAME) \
		--network host \
		- < docker/Dockerfile.user

.PHONY: launch_docker
launch_docker:
	$(DOCKER_RUN_CMD) --rm \
		$(DOCKER_INTERACTIVE_FLAGS) \
		-w $(CONTAINER_VOL) \
		-v $(HOST_VOL):$(CONTAINER_VOL) \
		-v ${HOME}:/mnt/${HOME} \
		-v /etc/timezone:/etc/timezone:ro \
		-v /etc/localtime:/etc/localtime:ro \
		--cap-add SYS_ADMIN \
		--cap-add SYS_TIME \
		-e NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES} \
		--shm-size=32gb \
		--security-opt apparmor=unconfined \
		--security-opt seccomp=unconfined \
		--name $(DOCKER_NAME) \
		-h $(DOCKER_NAME) \
		--add-host $(DOCKER_NAME):127.0.0.1 \
		-e HOST_HOSTNAME=$(HOSTNAME) \
		--user $(UID):$(GROUPID) \
		--net host \
		--device /dev/fuse \
		$(DOCKER_MOUNTS) \
		$(DOCKER_IMAGE_NAME):$(DOCKER_TAG) \
		$(DOCKER_COMMAND)

.PHONY: attach_docker
attach_docker:
	@$(MAKE) -f $(MAKEFILE_NAME) docker_add_user
	@$(MAKE) -f $(MAKEFILE_NAME) launch_docker

# Run docker
.PHONY: run_docker
run_docker:
	@$(MAKE) -f $(MAKEFILE_NAME) build_docker
	@$(MAKE) -f $(MAKEFILE_NAME) attach_docker || true

# Run docker with Waymo
.PHONY: run_waymo_docker
run_waymo_docker:
	@$(MAKE) -f $(MAKEFILE_NAME) build_docker
	@$(MAKE) -f $(MAKEFILE_NAME) build_waymo_docker
	@$(MAKE) -f $(MAKEFILE_NAME) attach_docker || true
