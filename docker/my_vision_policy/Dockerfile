FROM docker.io/library/ros:kilted-ros-core AS build

# Install pixi
RUN apt update && apt install -y git && curl -fsSL https://pixi.sh/install.sh | PIXI_VERSION=0.67.2 bash
ENV PATH="/root/.pixi/bin:$PATH"

# Add base dependencies
COPY aic_example_policies /ws_aic/src/aic/aic_example_policies
COPY aic_model /ws_aic/src/aic/aic_model
COPY aic_interfaces /ws_aic/src/aic/aic_interfaces
COPY aic_utils /ws_aic/src/aic/aic_utils
COPY pixi.toml pixi.lock robostack_mapping.json /ws_aic/src/aic/
COPY pixi_env_setup.sh /ws_aic/src/aic/

# Add custom policy package (includes weights/best.pt and all vision modules)
COPY my_vision_policy /ws_aic/src/aic/my_vision_policy

SHELL ["/bin/bash", "-c"]
RUN cd /ws_aic/src/aic && pixi install

WORKDIR /ws_aic/src/aic
COPY docker/my_vision_policy/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["--ros-args", "-p", "policy:=my_vision_policy.ros.VisionPolicy", "-p", "use_sim_time:=true"]
