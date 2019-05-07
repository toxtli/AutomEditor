FROM nvidia/cuda:10.1-base
MAINTAINER Carlos Toxtli (ctoxtli@gmail.com)
ENV DEBIAN_FRONTEND=noninteractive
RUN mkdir /content
WORKDIR /content
RUN apt update
RUN apt install -y git sudo wget unzip libboost-all-dev python-tk ffmpeg python-pip
RUN git clone https://github.com/TadasBaltrusaitis/OpenFace.git
RUN cd OpenFace
WORKDIR /content/OpenFace
RUN ./install.sh
RUN chmod +x download_models.sh
RUN ./download_models.sh
WORKDIR /content
RUN wget -O OpenFace/build/bin/model/patch_experts/cen_patches_0.25_of.dat https://www.dropbox.com/s/7na5qsjzz8yfoer/cen_patches_0.25_of.dat?dl=1
RUN wget -O OpenFace/build/bin/model/patch_experts/cen_patches_0.35_of.dat https://www.dropbox.com/s/k7bj804cyiu474t/cen_patches_0.35_of.dat?dl=1
RUN wget -O OpenFace/build/bin/model/patch_experts/cen_patches_0.50_of.dat https://www.dropbox.com/s/ixt4vkbmxgab1iu/cen_patches_0.50_of.dat?dl=1
RUN wget -O OpenFace/build/bin/model/patch_experts/cen_patches_1.00_of.dat https://www.dropbox.com/s/2t5t1sdpshzfhpj/cen_patches_1.00_of.dat?dl=1
RUN git clone https://github.com/naxingyu/opensmile.git
WORKDIR /content/opensmile
RUN sed -i '117s/(char)/(unsigned char)/g' src/include/core/vectorTransform.hpp
RUN ./buildStandalone.sh
WORKDIR /content
RUN git clone https://github.com/toxtli/AutomEditor.git
RUN pip install opencv-python --upgrade
RUN pip install git+https://github.com/rcmalli/keras-vggface.git
WORKDIR /content/AutomEditor/backend
RUN git pip install -r requirements.txt
WORKDIR /content/AutomEditor/backend/feature_extraction
RUN python features_from_file.py --sets ../../videos
WORKDIR /content
CMD ["date"]
EXPOSE 80
#RUN apt-get install -y nginx
#ENTRYPOINT ["/usr/sbin/nginx","-g","daemon off;"]
#ENTRYPOINT ["/bin/cat"]
#CMD ["/etc/passwd"]
#CMD ["ls","-al"]
