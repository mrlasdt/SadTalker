#!/usr/bin/env bash


# # If run from macOS, load defaults from webui-macos-env.sh
# if [[ "$OSTYPE" == "darwin"* ]]; then
#     export TORCH_COMMAND="pip install torch==1.12.1 torchvision==0.13.1"
# fi

# # python3 executable
if [[ -z "${python_cmd}" ]]
then
    python_cmd="python3"
fi

# # git executable
# if [[ -z "${GIT}" ]]
# then
#     export GIT="git"
# fi

# # python3 venv without trailing slash (defaults to ${install_dir}/${clone_dir}/venv)
# if [[ -z "${venv_dir}" ]]
# then
#     venv_dir="venv"
# fi

if [[ -z "${LAUNCH_SCRIPT}" ]]
then
    LAUNCH_SCRIPT="launcher.py"
fi

# # this script cannot be run as root by default
# can_run_as_root=1

# # read any command line flags to the webui.sh script
# while getopts "f" flag > /dev/null 2>&1
# do
#     case ${flag} in
#         f) can_run_as_root=1;;
#         *) break;;
#     esac
# done

# # Disable sentry logging
# export ERROR_REPORTING=FALSE

# # Do not reinstall existing pip packages on Debian/Ubuntu
# export PIP_IGNORE_INSTALLED=0

# # Pretty print
delimiter="################################################################"

# printf "\n%s\n" "${delimiter}"
# printf "\e[1m\e[32mInstall script for SadTalker + Web UI\n"
# printf "\e[1m\e[34mTested on Debian 11 (Bullseye)\e[0m"
# printf "\n%s\n" "${delimiter}"

# # Do not run as root
# if [[ $(id -u) -eq 0 && can_run_as_root -eq 0 ]]
# then
#     printf "\n%s\n" "${delimiter}"
#     printf "\e[1m\e[31mERROR: This script must not be launched as root, aborting...\e[0m"
#     printf "\n%s\n" "${delimiter}"
#     exit 1
# else
#     printf "\n%s\n" "${delimiter}"
#     printf "Running on \e[1m\e[32m%s\e[0m user" "$(whoami)"
#     printf "\n%s\n" "${delimiter}"
# fi

# if [[ -d .git ]]
# then
#     printf "\n%s\n" "${delimiter}"
#     printf "Repo already cloned, using it as install directory"
#     printf "\n%s\n" "${delimiter}"
#     install_dir="${PWD}/../"
#     clone_dir="${PWD##*/}"
# fi

# # Check prerequisites
# gpu_info=$(lspci 2>/dev/null | grep VGA)
# case "$gpu_info" in
#     *"Navi 1"*|*"Navi 2"*) export HSA_OVERRIDE_GFX_VERSION=10.3.0
#     ;;
#     *"Renoir"*) export HSA_OVERRIDE_GFX_VERSION=9.0.0
#         printf "\n%s\n" "${delimiter}"
#         printf "Experimental support for Renoir: make sure to have at least 4GB of VRAM and 10GB of RAM or enable cpu mode: --use-cpu all --no-half"
#         printf "\n%s\n" "${delimiter}"
#     ;;
#     *) 
#     ;;
# esac
# if echo "$gpu_info" | grep -q "AMD" && [[ -z "${TORCH_COMMAND}" ]]
# then
#     export TORCH_COMMAND="pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/rocm5.2"
# fi  

# for preq in "${GIT}" "${python_cmd}"
# do
#     if ! hash "${preq}" &>/dev/null
#     then
#         printf "\n%s\n" "${delimiter}"
#         printf "\e[1m\e[31mERROR: %s is not installed, aborting...\e[0m" "${preq}"
#         printf "\n%s\n" "${delimiter}"
#         exit 1
#     fi
# done

# if ! "${python_cmd}" -c "import venv" &>/dev/null
# then
#     printf "\n%s\n" "${delimiter}"
#     printf "\e[1m\e[31mERROR: python3-venv is not installed, aborting...\e[0m"
#     printf "\n%s\n" "${delimiter}"
#     exit 1
# fi

# printf "\n%s\n" "${delimiter}"
# printf "Create and activate python venv"
# printf "\n%s\n" "${delimiter}"
# cd "${install_dir}"/"${clone_dir}"/ || { printf "\e[1m\e[31mERROR: Can't cd to %s/%s/, aborting...\e[0m" "${install_dir}" "${clone_dir}"; exit 1; }
# if [[ ! -d "${venv_dir}" ]]
# then
#     "${python_cmd}" -m venv "${venv_dir}"
#     first_launch=1
# fi
# # shellcheck source=/dev/null
# if [[ -f "${venv_dir}"/bin/activate ]]
# then
#     source "${venv_dir}"/bin/activate
# else
#     printf "\n%s\n" "${delimiter}"
#     printf "\e[1m\e[31mERROR: Cannot activate python venv, aborting...\e[0m"
#     printf "\n%s\n" "${delimiter}"
#     exit 1
# fi

printf "\n%s\n" "${delimiter}"
printf "Launching launcher.py..."
printf "\n%s\n" "${delimiter}"      
exec "${python_cmd}" "${LAUNCH_SCRIPT}" "$@"