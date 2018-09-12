clear; clc;
warning off all
root=cd; addpath(genpath([root '/DATA'])); addpath(genpath([root '/FUN']));

%load fMRI176x200; data1 = fMRImciNc; clear fMRImciNc;
%load fMRI_h; data1 = fMRImciNc; clear fMRImciNc;
load fMRI80; data1 = fMRImciNc; clear fMRImciNc;
nSubj=length(lab);
nROI=size(data1{1},2);
nDegree=size(data1{1},1);
label=input('PSR[1],SR[2],PSRTL[3],SRTL[4]:');

if label==1
    %lambda=0;
    lambda=0:0.025:0.495;
    lambda=[lambda 0.495]
    disp('Press any key:'); pause;
    nL=length(lambda);
    brainNetSet=cell(1,nL);
    
    for iL=1:nL
        
        brainNet=zeros(nROI,nROI,nSubj);
        for i=1:nSubj
            tmp=data1{i}(:,1:nROI);%%%%%%%%%%%%%%%%%%%%%%%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            tmp=tmp-repmat(mean(tmp)',1,nDegree)';% centrlization
            
            currentNet = accel_grad_PSR_cavin(tmp,lambda(iL));
            currentNet=currentNet-diag(diag(currentNet));
            brainNet(:,:,i)=currentNet;
        end
        brainNetSet{iL}=brainNet;
        fprintf('Done lambda=%d networks!\n',iL);
        
    end
    save('brainNetSet_PSR_NITRC.mat','brainNetSet','lab');
end


%% Network learning based on sparse representation(SR) - SLEP
if label==2
    %Parameter setting for SLEP
    lambda=0:0.025:0.495;
    lambda=[lambda 0.495]
    disp('Press any key:'); pause;
    nPar=length(lambda);
    brainNetSet=cell(1,nPar);
    
    opts=[];
    opts.init=2;% Starting point: starting from a zero point here
    opts.tFlag=0;% termination criterion
    % abs( funVal(i)- funVal(i-1) ) ¡Ü .tol=10e?4 (default)
    %For the tFlag parameter which has 6 different termination criterion.
    % 0 ? abs( funVal(i)- funVal(i-1) ) ¡Ü .tol.
    % 1 ? abs( funVal(i)- funVal(i-1) ) ¡Ü .tol ¡Á max(funVal(i),1).
    % 2 ? funVal(i) ¡Ü .tol.
    % 3 ? kxi ? xi?1k2 ¡Ü .tol.
    % 4 ? kxi ? xi?1k2 ¡Ü .tol ¡Á max(||xi||_2, 1).
    % 5 ? Run the code for .maxIter iterations.
    opts.nFlag=0;% normalization option: 0-without normalization
    opts.rFlag=0;% regularization % the input parameter 'rho' is a ratio in (0, 1)
    opts.rsL2=0; % the squared two norm term in min  1/2 || A x - y||^2 + 1/2 rsL2 * ||x||_2^2 + z * ||x||_1
    fprintf('\n mFlag=0, lFlag=0 \n');
    opts.mFlag=0;% treating it as compositive function
    opts.lFlag=0;% Nemirovski's line search
    
    for L=1:nPar
        brainNet=zeros(nROI,nROI,nSubj);
        for i=1:nSubj
            tmp=data1{i};
            tmp=tmp-repmat(mean(tmp')',1,nROI);% data centralization
            currentNet=zeros(nROI,nROI);
            for j=1:nROI
                y=[tmp(:,j)];
                A=[tmp(:,setdiff(1:nROI,j))];
                [x, funVal1, ValueL1]= LeastR(A, y, lambda(L), opts);
                currentNet(setdiff(1:nROI,j),j) = x;
            end
            brainNet(:,:,i)=currentNet;
        end
        brainNetSet{L}=brainNet;
        fprintf('Done %d/%d networks!\n',L,nPar);
    end
    save('brainNetSet_SR_NITRC.mat','brainNetSet','lab');
end

if label==3
    
    lambda=0:0.025:0.495;
    lambda=[lambda 0.495]
    
    disp('Press any key:'); pause;
    nL=length(lambda);
    brainNetSet=cell(1,nL);
    load('brainNet_PC_HCP.mat')
    clear brainNet_PC
    WH=mean_brainNet_PC(1:90,1:90);
    for iL=1:nL;
        
        brainNet=zeros(nROI,nROI,nSubj);
        for i=1:nSubj
            tmp=data1{i}(:,1:nROI);%%%%%%%%%%%%%%%%%%%%%%%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            tmp=tmp-repmat(mean(tmp)',1,nDegree)';% centrlization
            Maxt=max(max(tmp));
            Mint=min(min(tmp));
            tmp=tmp./(Maxt-Mint);%normalized
            currentNet = accel_grad_PSR_hub(tmp,lambda(iL),WH);
            currentNet = (currentNet + currentNet')/2;
            currentNet=currentNet-diag(diag(currentNet));
            brainNet(:,:,i)=currentNet;
        end
        brainNetSet{iL}=brainNet;
        fprintf('Done lambda=%d networks!\n',iL);
        
    end
    save('brainNetSet_PSRTL_NITRC2.mat','brainNetSet','lab');
end

if label==4
    %Parameter setting for SLEP
    lambda=0:0.025:0.495;
    lambda=[lambda 0.495]
    disp('Press any key:'); pause;
    nPar=length(lambda);
    brainNetSet=cell(1,nPar);
    load('brainNet_PC_HCP.mat')
    clear brainNet_PC
    WH=mean_brainNet_PC(1:90,1:90);
    opts=[];
    opts.init=2;% Starting point: starting from a zero point here
    opts.tFlag=0;% termination criterion
    % abs( funVal(i)- funVal(i-1) ) ¡Ü .tol=10e?4 (default)
    %For the tFlag parameter which has 6 different termination criterion.
    % 0 ? abs( funVal(i)- funVal(i-1) ) ¡Ü .tol.
    % 1 ? abs( funVal(i)- funVal(i-1) ) ¡Ü .tol ¡Á max(funVal(i),1).
    % 2 ? funVal(i) ¡Ü .tol.
    % 3 ? kxi ? xi?1k2 ¡Ü .tol.
    % 4 ? kxi ? xi?1k2 ¡Ü .tol ¡Á max(||xi||_2, 1).
    % 5 ? Run the code for .maxIter iterations.
    opts.nFlag=0;% normalization option: 0-without normalization
    opts.rFlag=0;% regularization % the input parameter 'rho' is a ratio in (0, 1)
    opts.rsL2=0; % the squared two norm term in min  1/2 || A x - y||^2 + 1/2 rsL2 * ||x||_2^2 + z * ||x||_1
    fprintf('\n mFlag=0, lFlag=0 \n');
    opts.mFlag=0;% treating it as compositive function
    opts.lFlag=0;% Nemirovski's line search
    
    for L=1:nPar
        brainNet=zeros(nROI,nROI,nSubj);
        for i=1:nSubj
            tmp=data1{i};
            tmp=tmp-repmat(mean(tmp')',1,nROI);% data centralization
            currentNet=zeros(nROI,nROI);
            for j=1:nROI
                y=[tmp(:,j)];
                A=[tmp(:,setdiff(1:nROI,j))];
                [x, funVal1, ValueL1]= LeastR_TL(A, y, lambda(L),WH(setdiff(1:nROI,j),j), opts);
                currentNet(setdiff(1:nROI,j),j) = x;
            end
            brainNet(:,:,i)=currentNet;
        end
        brainNetSet{L}=brainNet;
        fprintf('Done %d/%d networks!\n',L,nPar);
    end
    save('brainNetSet_SRTL_NITRC2.mat','brainNetSet','lab');
end
