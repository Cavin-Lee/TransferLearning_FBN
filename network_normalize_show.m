clear
clc

load('brainNetSet_PSRTL_NITRC.mat')
brain=brainNetSet{11};% lambda=0.5
mean_brain=mean(brain,3);
mean_brain=(mean_brain+mean_brain')/2;
maxcon=max(max(mean_brain));
mincon=min(min(mean_brain));

mean_brain(mean_brain>0)=mean_brain(mean_brain>0)/maxcon;
mean_brain(mean_brain<0)=-1*mean_brain(mean_brain<0)/mincon;

imagesc(mean_brain)

axis off