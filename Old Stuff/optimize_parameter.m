

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PARAMETER TO BE OPTIMIZED %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


clear
clc
diary on

skip_samples_train = 200;   % Number of samples which are skipped when  
                            % generating ringpuffersequences for HMM-Training
                            
skip_samples_test = 100;    % Number of samples which are skipped when  
                            % generating ringpuffersequences for
                            % precision/recall calculation
                            
thresholds = 0:0.02:1;       % Certainty threshold vector:
                            % Precision/recall will be calculated for each
                            % certainty threshold. 
                            % The certainty has to be higher than the
                            % threshold in order to classify the input as
                            % SPU/SPD. If its lower it will be classified
                            % as FIX
                            
sequencelengths = 300 ;     % Vector with sequencelengths:
                            % HMM-Training, classifier generation and 
                            % precision/recall calculation will be performed
                            % for each sequence in this vector
                            
continue_optimization=1;    % When set to 1 a existing results.mat will be 
                            % loaded and the permutation will be used for 
                            % further optimization
                            % New results will be added at the end of the
                            % result cell
                            
leave_one_out = 1;          % When set to 1 training and testing will be 
                            % performed for each subject.
                            % When set to 0 only training and generation of
                            % the classifier will be performed
                            
cutoff = 0;
                            

%%

%%%%%%%%%%%%%
% CONSTANTS %
%%%%%%%%%%%%%

%% Study Design

    g_h=720;    % [px]: Gaze height in 
    h=300;      % [mm]: Video height in
    %d=680;      % [mm]: Distance eyetracker - display
    d=540;
    g_w=1280;
    w=300/9*16;
    % Namespace:
    % - Subcluster = timespan with 1 Fix, 1 SPU or 1 SPD
    % - Cluster = consists of 1 Fix and 1 SPU or SPD part
    % - Episode = consists of 8 cluster
    % - Video = consists of 6 episodes and 5 shaun parts
    % - sequencetype = SPU,SPD,FIX,Shaun

    NUM_VIDEOS = 2;
    NUM_EPISODES = 6;
    NUM_CLUSTER = 8;

    DURATION_CLUSTER = 4.067;   % [s]
    DURATION_FIX = 2.533;       % [s]
    DURATION_SHAUN = 30.033;    % [s]

    % permut indicates which cluster (columnwise) in which episode (rowwise) has
    % SPU (1) or SPD(1)
    % Example: first video of subject 02 has following episodes CDEFGH. 
    % This means that row 3 indicates the cluster in the first episode (C) row
    % 4 in the second (D) and so on.
    permut = [1 1 1 0 0 0 0 1;...
              1 1 0 1 0 1 0 0;...
              0 0 1 0 0 1 1 1;...
              1 1 0 0 1 0 1 0;...
              0 1 1 1 0 0 1 0;...
              1 1 0 1 0 0 1 0;...
              0 1 1 0 1 0 0 1;...
              0 1 1 1 0 0 0 1;...
              0 1 1 0 1 0 1 0;...
              0 1 0 1 0 0 1 1;...
              0 1 0 1 0 1 0 1;...
              1 1 0 1 0 1 0 0];
      
%% Signal Processing

    react_time = 0.05;  % [s]: When labeling and splitting each subcluster  
                        % this constant will be added to take into account  
                        % the reaction delay
                        
    % Saccade detection:
    v_min = 100;        % [°/s]: min velocity to detect a saccade
    v_max = 30;         % [°/s]: max velocity to classify as fixation 
    phi_min = 1.5;      % [°]: min angle to be passed when velocity > v_max
    max_skip = 2;       % # samples (< v_max) to be ignored
    
    % Filter
    fpass = 0.01;       % ]0;1[: pass frequency (for further explanation: "doc designfilt")
    fstop = 0.05;       % ]0;1[: stop frequency
    apass = 1;          % [dB]: pass attenuation
    astop = 15;         % [dB]: stop attenuation
    

%% HMM Training

    addpath(genpath('./HMMall'))    % include HMM Toolkit

    M = 1;                          % Number of gauss mixtures
    Q = 4;                          % Number of states
    max_iter = 15;                  % Max number of iterations
    tol = 5e-4;                     % Tolerance
    O = 1;                          % Number of values for each time sample
    cov_type = 'full'; 

 
    %%    
    
%%%%%%%%%%%%%%%%
% LOAD & LABEL %
%%%%%%%%%%%%%%%%

%% Load Data

disp('Load results');
for i=1:7
    data{i,1}=load_data(['Results/part0' num2str(i-1) '/set0.coord']);
    data{i,2}=load_data(['Results/part0' num2str(i-1) '/set1.coord']);
end

for i=8:9
    data{i,1}=load_data(['Results/part0' num2str(i) '/set0.coord']);
    data{i,2}=load_data(['Results/part0' num2str(i) '/set1.coord']);
end 

%% Convert timestamps to seconds and transform to degrees
   
   % data = cellfun(@(x) [10^-6*x(:,1) atand(((1 - (2* (x(:,3)/g_h)) ))*h/(2*d)) x(:,3:4)],data,'UniformOutput',false);
%x(:,3) ist y koordinate--> wieso entsprechenden Winkel an x position
%x(:,2) schreiben?

data = cellfun(@(x) [10^-6*x(:,1) atand(((1 - (2*(x(:,2)/g_w))))*w/(2*d)) atand(((1 - (2* (x(:,3)/g_h)) ))*h/(2*d)) x(:,4)],data,'UniformOutput',false);

    
%% Label each cluster and split

    disp('Label each subcluster and split');

    DURATION_EPISODE = NUM_CLUSTER*DURATION_CLUSTER;
    DURATION_SP = DURATION_CLUSTER - DURATION_FIX;
    DURATION_E_S = DURATION_EPISODE + DURATION_SHAUN;
    
    fix = cell(size(data,1),NUM_VIDEOS*NUM_EPISODES*NUM_CLUSTER);
    spu = cell(size(data,1),NUM_VIDEOS*NUM_EPISODES*NUM_CLUSTER/2);
    spd = cell(size(data,1),NUM_VIDEOS*NUM_EPISODES*NUM_CLUSTER/2);
    % shaun = cell(size(data,1),(NUM_EPISODES-1)*NUM_VIDEOS);

    for i=1:size(data,1)        % Number of files
        for v = 1:NUM_VIDEOS        
            for j=1:NUM_EPISODES           
                num_spu = 1;
                num_spd = 1;
                for k=1:NUM_CLUSTER
                    % Fixation is always the first DURATION_CLUSTER-DURATION_SP of the cluster
                    fix{i,(v-1)*NUM_CLUSTER*NUM_EPISODES + (j-1)*NUM_CLUSTER+k} = ...
                        data{i,v}( (data{i,v}(:,1) >= ((j-1) * DURATION_E_S + (k - 1) * DURATION_CLUSTER + react_time )) ...
                                 & (data{i,v}(:,1) <= ((j-1) * DURATION_E_S + k * DURATION_CLUSTER - DURATION_SP + react_time )) ,:);

                    % Decide if last DURATION_SP of the cluster is SPU or SPD
                    if permut(mod(i+(v-1)*6+j-2,12)+1,k) == 1
                        % SPU
                        spu{i,(v-1)*NUM_CLUSTER*NUM_EPISODES/2 + (j-1)*NUM_CLUSTER/2+num_spu} = ...
                            data{i,v}( (data{i,v}(:,1) >= ((j-1) * DURATION_E_S + k * DURATION_CLUSTER - DURATION_SP + react_time )) ...
                                     & (data{i,v}(:,1) <= ((j-1) * DURATION_E_S + k * DURATION_CLUSTER + react_time )) ,:);
                        num_spu = num_spu + 1;
                    else
                        % SPD
                        spd{i,(v-1)*NUM_CLUSTER*NUM_EPISODES/2 + (j-1)*NUM_CLUSTER/2+num_spd} = ...
                            data{i,v}( (data{i,v}(:,1) >= ((j-1) * DURATION_E_S + k * DURATION_CLUSTER - DURATION_SP + react_time )) ...
                                     & (data{i,v}(:,1) <= ((j-1) * DURATION_E_S + k * DURATION_CLUSTER + react_time )) ,:);
                        num_spd = num_spd + 1;
                    end
                end
                % Shaun is always between the episodes
                 if j < NUM_EPISODES
                     shaun{i,j+(v-1)*(NUM_EPISODES-1)}= data{i,v}( (data{i,v}(:,1) >= ((j-1) * DURATION_E_S + DURATION_EPISODE + react_time  )) ...
                                                                 & (data{i,v}(:,1) <= ( j * DURATION_E_S + react_time  )) ,:);
                 end
            end
        end
    end
    %%

%%%%%%%%%%%%%%%%%%%%
% PRE-PROCESS DATA %
%%%%%%%%%%%%%%%%%%%%

%% New PCA Stuff
spd_d =cellfun(@(x) downsample(x,15),spd,'UniformOutput',false); % only take every 16. value: 1000Hz->62.5Hz
%spd_d{u,c}(:,1)=spd_d{u,c}(:,1).*100; %better PC1?

u=1;  %current User
c=3;  %current Cluster

t_m=mean(spd_d{u,c}(:,1));
x_m=mean(spd_d{u,c}(:,2));
y_m=mean(spd_d{u,c}(:,3));

%Plot input data

figure('Name','Input Data')
scatter3(spd_d{u,c}(:,1),spd_d{u,c}(:,2),spd_d{u,c}(:,3),'r');
xlabel('Time'),ylabel('X'),zlabel('Y')
[coeff,score,latent]=pca((spd_d{u,c}(:,1:3)));
hold on
quiver3(t_m,x_m,y_m,coeff(1,1),coeff(2,1),coeff(3,1),'b');
quiver3(t_m,x_m,y_m,coeff(1,2),coeff(2,2),coeff(3,2),'o')
quiver3(t_m,x_m,y_m,coeff(1,3),coeff(2,3),coeff(3,3),'y')
hold off

%Project points in pc1 direction
% spd_d_new=(coeff'*(spd_d{u,c}(:,1:3))')';

% vec1=coeff(:,1);
% vec1(2)=0;
% t_axis=[1;0;0];
% a_x=dot(vec1,t_axis);
% 
% vec2=coeff(:,1);
% vec2(3)=0;
% a_y=dot(vec2,t_axis);
% 
% Rx=eye(3,3);
% Rx(2,2)=cos(a_x);
% Rx(3,3)=cos(a_x);
% 
% Rx(2,3)=-sin(a_x);
% Rx(3,2)=sin(a_x);
% test=Rx'*vec1;
% 
% 
% Ry=eye(3,3);
% Ry(1,1)=cos(a_x);
% Rx(3,3)=cos(a_x);
% 
% Rx(2,3)=-sin(a_x);
% Rx(3,2)=sin(a_x);
% 
% quiver3(0,0,0,vec1(1),vec1(2),vec1(3),'r');
% xlabel('Time'),ylabel('X'),zlabel('Y')
% hold on
% quiver3(0,0,0,test(1),test(2),test(3),'b');
% quiver3(0,0,0,coeff(1,1),coeff(2,1),coeff(3,1),'g');




% %%test stuff on shaun
% figure
% shaun_d =cellfun(@(x) downsample(x,15),shaun,'UniformOutput',false); % only take every 16. value: 1000Hz->62.5Hz
% shaun_d{1,1}(:,1)=shaun_d{1,1}(:,1).*100; %better PC1
% scatter3(shaun_d{1,1}(:,1),shaun_d{1,1}(:,2),shaun_d{1,1}(:,3),'r');
% xlabel('Time'),ylabel('X'),zlabel('Y')
% coeff_shaun=pca((shaun_d{1,1}(:,1:3)));
% shaun_new=(coeff_shaun'*(shaun_d{1,1}(:,1:3))')';
% figure
% scatter3(shaun_new(:,1),shaun_new(:,2),shaun_new(:,3),'m','.');
% xlabel('Time'),ylabel('X'),zlabel('Y')

%Visualize new PCA(hopefully original kartesian system)
% t_m2=mean(spd_d_new(:,1));
% x_m2=mean(spd_d_new(:,2));
% y_m2=mean(spd_d_new(:,3));
% coeff2=pca((spd_d_new(:,1:3)));
% 
% %Plot projected data
% figure
% scatter3(spd_d_new(:,1),spd_d_new(:,2),spd_d_new(:,3),'m','.');
% xlabel('Time'),ylabel('X'),zlabel('Y')
% hold on
% quiver3(t_m2,x_m2,y_m2,coeff2(1,1),coeff2(2,1),coeff2(3,1),'b');
% quiver3(t_m2,x_m2,y_m2,coeff2(1,2),coeff2(2,2),coeff2(3,2),'o')
% quiver3(t_m2,x_m2,y_m2,coeff2(1,3),coeff2(2,3),coeff2(3,3),'y')
% hold off

%axang=rotm2axang(coeff(:,1));
%Pursuit turned--> sehr kleine x,y werte wenn auf z Achse ausgerichtet:
%Genau was wir wollen?



%%Fixation detection(simple approch)
%Check if x distance of every point to mean is small 
% r=0.2;
% for i=1:size(spd_d_new,1)
%     if abs(x_m2-spd_d_new(i,2))<r
%         spd_d_new(i,4)=1;
%     else
%         spd_d_new(i,4)=0;
%     end
% end
% 
% %Check for saccacedes
% for j=2:size(spd_d_new,1)
%     if spd_d_new(i,4)==1;
%         if (sum(abs(spd_d{1,1}(j,2:3)-spd_d{1,1}(j-1,2:3)))>2 || (sum(abs(spd_d{1,1}(j,2:3)-spd_d{1,1}(j-1,2:3)))<0.05))
%             spd_d_new(i,4)=0;
%         end
%     end
% end
% figure
% scatter3(spd_d_new(spd_d_new(:,4)==1,1),spd_d_new(spd_d_new(:,4)==1,2),spd_d_new(spd_d_new(:,4)==1,3),'g','.');
% xlabel('Time'),ylabel('X'),zlabel('Y')
% hold on
% scatter3(spd_d_new(spd_d_new(:,4)==0,1),spd_d_new(spd_d_new(:,4)==0,2),spd_d_new(spd_d_new(:,4)==0,3),'r','.');
% r=0.2;
% teta=-pi:0.01:pi;
% x_c=r*cos(teta);
% y_c=r*sin(teta);
% plot3(t_m2+zeros(1,numel(x_c)),x_m2+y_c,y_m2+x_c);
% 
% %rectangle
% hold off


%%Window 
window_size=15;
teta=-pi:0.01:pi;
k=1;
while(k<=size(spd_d{u,c},1)-window_size+1)
    saccade=false;
    d=zeros(2);
    window=spd_d{u,c}(k:k+window_size-1,1:3);
    for i=2:window_size   
        %d(i-1)=pdist([window(i,2:3);window(i-1,2:3)]);   
        d(i-1,:)=window(i,2:3)-window(i-1,2:3);
        %Check for saccade
        if dot(d(i-1,:),d(i-1,:))>0.64
            disp('\n Saccade detected!');
            saccade=true;
            break
        end
    end
    if saccade==false
        
        %median
        average_distance=mean(d);    

        %Visualization
%         scatter3(window(:,1),window(:,2),window(:,3));
%         xlabel('Time'),ylabel('X'),zlabel('Y')
%        % 2.4-7.2 °/window -->0.16-0.48 °/frame
%         figure
%         scatter(d(:,1),d(:,2));
%         axis equal
%         hold on
%         scatter(average_distance(1),average_distance(2));
%         r=0.23;    
%         x_c=r*cos(teta);
%         y_c=r*sin(teta);
%         plot(average_distance(1)+x_c,average_distance(2)+y_c);
%         
%         r=2.6e-4;    
%         x_c=r*cos(teta);
%         y_c=r*sin(teta);
%         plot(average_distance(1)+x_c,average_distance(2)+y_c);
    
       
        dist=dot(average_distance,average_distance);
        
        
        %mean
        if dist<0.0625 && dist>0.01
            disp(' Smooth Pursuit!!!!');
        elseif dist <0.01
            disp(' Fixation!!!!');
        else
            disp(' Saccade? or whatever');
        end
    end
    k=k+1;
end
saccade_log=cell(9,48);
for i=1:9
    
        disp(sprintf('\nUser:%d',i));
        smooth_detection(spd,i,1);
        saccade_log{i,1}=saccade_l;
    
end
minimal_saccade=cellfun(@(x) min(x),saccade_log,'UniformOutput',false); 
maximum_saccade=cellfun(@(x) max(x),saccade_log,'UniformOutput',false); 



    %% Construct Filter
    
%     filterr = designfilt('lowpassfir','PassbandFrequency',fpass,...
%            'StopbandFrequency',fstop,'PassbandRipple',apass,'StopbandAttenuation',astop);
%      filter_coeff = filterr.Coefficients;
%     %  fvtool(filterr) 
%     order=filtord(filterr);      
%     
%     %% Detect & Split Saccades
%  
% 
%     disp('Detect & Split Saccades (FIX)');
%     fix = cellfun(@(x) detect_saccade(x,v_min,v_max,phi_min,max_skip),fix,'UniformOutput',false);
%     fix_split = cellfun(@(x) split_saccade(x,order),fix,'UniformOutput',false);
% 
%     disp('Detect & Split Saccades (SPU)');
%     spu = cellfun(@(x) detect_saccade(x,v_min,v_max,phi_min,max_skip),spu,'UniformOutput',false);
%     spu_split = cellfun(@(x) split_saccade(x,order),spu,'UniformOutput',false);
% 
%     disp('Detect & Split Saccades (SPD)');
%     spd = cellfun(@(x) detect_saccade(x,v_min,v_max,phi_min,max_skip),spd,'UniformOutput',false);
%     spd_split = cellfun(@(x) split_saccade(x,order),spd,'UniformOutput',false);
% 
% 
%     %% Reorder
%     
%     fix_split_ord_subj = reorder_cell_subject(fix_split);
%     spu_split_ord_subj = reorder_cell_subject(spu_split);
%     spd_split_ord_subj = reorder_cell_subject(spd_split);
% 
%     %% Generate config for save
% 
%     if continue_optimization
%         load('results.mat')
%         cnt=size(result,1);
%     else
%         cnt=1;
%         result={'Subject' , 'Video' , 'Sequencelength' , 'Cutoff', 'FilterCoeff', 'Thresholds' , 'Pre_SPU' , 'Pre_SPD' , 'Rec_SPU' , 'Rec_SPD' , 'prior' , 'transmat' , 'mu' , 'Sigma' , 'mixmat' , 'mean_class_ma' , 'cov' , 'skip_samples_train', 'skip_samples_test', 'Perm', 'config'};
%     end
%     
%     config={'react_time' , 'v_min1', 'v_min2', 'phi_min', 'fpass', 'fstop', 'apass', 'astop', 'apass', 'M', 'Q', 'max_iter', 'thresh'};
%     config(2,:)={react_time , v_min, v_max, phi_min, fpass, fstop, apass, astop, apass, M, Q, max_iter, tol};
% 
%     %%
% 
% %%%%%%%%%%%%%%%%%%%%%%
% % BEGIN OPTIMIZATION %
% %%%%%%%%%%%%%%%%%%%%%%
% 
% for sequencelength=sequencelengths
%     if leave_one_out
%         for subject=1:size(fix_split_ord_subj,1) % determine which subject to use for testing (rest for training and generating classifier)
% 
%         %% Generate Sequences
% 
%             disp(['Generate Sequences for testsubject ' num2str(subject)]);
% 
%             % Dont include subject -> [1:subject-1 subject+1:9]    
%             % Perm determines which sequences will be used for HMM training and
%             % which sequences will be used for generation of the classifier
%             [train_spu, val_spu, Perm{1}]=generate_random_ringpuffseq(spu_split_ord_subj([1:subject-1 subject+1:size(fix_split_ord_subj,1)],:),sequencelength,skip_samples_train,filter_coeff,g_h,h,d,cutoff);
%             [train_spd, val_spd, Perm{2}]=generate_random_ringpuffseq(spd_split_ord_subj([1:subject-1 subject+1:size(fix_split_ord_subj,1)],:),sequencelength,skip_samples_train,filter_coeff,g_h,h,d,cutoff);
%             [train_fix, val_fix, Perm{3}]=generate_random_ringpuffseq(fix_split_ord_subj([1:subject-1 subject+1:size(fix_split_ord_subj,1)],:),sequencelength,skip_samples_train,filter_coeff,g_h,h,d,cutoff);
% 
%             %% Train HMMs
% 
%                % initial guess of parameters
%                 prior0 = normalise(rand(Q,1)); % prior(i) = Pr(Q(1) = i), --> Einsprungvektor
%                 transmat0 = mk_stochastic(rand(Q,Q)); %transmat(i,j) = Pr(Q(t+1)=j | Q(t)=i) --> Transitionmatrix
%                 %[mu0, Sigma0] = mixgauss_init(Q*M, train_spd{1}, cov_type);
%                 [mu0, Sigma0] = mixgauss_init(Q*M, train_spd{1}, cov_type);
%                 mu0 = reshape(mu0, [O Q M]);
%                 Sigma0 = reshape(Sigma0, [O O Q M]);
%                 %Sigma0= 0.4*ones(1,1,4);
%                 mixmat0 = ones(4,1);
%                 % Notation: Q(t) = hidden state, Y(t) = observation, M(t) = mixture variable
%             disp('Train HMMs');
%             disp(['SPU: ' num2str(length(train_spu)) '  SPD: ' num2str(length(train_spd)) '  FIX: ' num2str(length(train_fix)) ]);
% 
%             tic;
%             [~, prior{1}, transmat{1}, mu{1}, Sigma{1}, mixmat{1}] = ...
%                 mhmm_em(train_spu, prior0, transmat0, mu0, Sigma0, mixmat0,'max_iter',max_iter,'thresh',tol);
%             toc;
%             tic;
%             disp(sprintf('\tSPU-HMM finished'));
%             [~, prior{2}, transmat{2}, mu{2}, Sigma{2}, mixmat{2}] = ...
%                 mhmm_em(train_spd, prior0, transmat0, mu0, Sigma0, mixmat0,'max_iter',max_iter,'thresh',tol);
%             toc;
%             tic;
%             disp(sprintf('\tSPD-HMM finished'));
%             [~, prior{3}, transmat{3}, mu{3}, Sigma{3}, mixmat{3}] = ...
%                 mhmm_em(train_fix, prior0, transmat0, mu0, Sigma0, mixmat0,'max_iter',max_iter,'thresh',tol);
%             toc;
%             disp(sprintf('\tFIX-HMM finished'));
% 
%         %% Generate Mahalanobis Classifier
% 
%             disp('Generate Mahalanobis Classifier (FIX)');
%             [mean_class_ma{1}, ~ , cov{1}]=generate_classifier(val_spu, prior, transmat, mu, Sigma, mixmat,'ma');
%             disp('Generate Mahalanobis Classifier (SPU)');
%             [mean_class_ma{2}, ~ , cov{2}]=generate_classifier(val_spd, prior, transmat, mu, Sigma, mixmat,'ma');
%             disp('Generate Mahalanobis Classifier (SPD)');
%             [mean_class_ma{3}, ~ , cov{3}]=generate_classifier(val_fix, prior, transmat, mu, Sigma, mixmat,'ma');
% 
%         %% Meassure Distance and classify test data (Mahalanobis)
% 
%             for vid = 1:2
%                 disp(['Meassure Distance and classify test data (Mahalanobis): Video ' num2str(vid)]);
%                 testdata=data{subject,vid}(:,[1 2])';        
%               %  testdata_split_seq{1,1} = diff(testdata(order+1:end,6))';
%              %   testdata_split_seq = testdata(:,[1 2])';
% 
%                 true_pos_SPU(1:length(thresholds)) = 0;
%                 true_pos_SPD(1:length(thresholds)) = 0;
%                 false_neg_SPU(1:length(thresholds)) = 0;
%                 false_neg_SPD(1:length(thresholds)) = 0;
%                 false_pos_SPU(1:length(thresholds)) = 0;
%                 false_pos_SPD(1:length(thresholds)) = 0;
% 
%                 for j=1:skip_samples_test:size(testdata,2)-sequencelength
% 
%                     % extract ring 
%                     data_ring=testdata(:,j:j+sequencelength-1);
%                     
%                     % filter ring
%                     tmp = conv(data_ring(2,:),filter_coeff);
%                     tmp = (1-2*d/h*tand(tmp))*g_h/2;
%                     if cutoff
%                         data_test_ring=diff(tmp((length(filter_coeff)+1)/2:end-(length(filter_coeff)-1)/2));
%                     else
%                         data_test_ring=diff(tmp);
%                     end
% 
%                     [type, percentage]=get_sequencetype(subject,vid,max([data_ring(1,1)-react_time 0]),max([data_ring(1,end)-react_time data_ring(1,end)-data_ring(1,1)]));
% 
%                     % Consider only sequences whose sequencetype is not shaun and
%                     % are not part of 2 sequencetypes
%                     if percentage == 1 && type ~= 4
%                         % classify ring only if necessary
%                         distance_ring=classify_data(data_test_ring,mean_class_ma,cov, prior, transmat, mu, Sigma, mixmat,'ma');
%                         % Calculate certainty (compare the 2 lowest distances)
%                         temp = distance_ring(1:end-1);
%                         temp = sort(temp);
%                         certainty = (temp(2)/sum(temp(1:2))-0.5)*2;
% 
%                         for k = 1:length(thresholds)
%                             if certainty < thresholds(k)
%                                 % if decision is too uncertain then classify as
%                                 % FIX
%                                 distance_ring(end,1)=3;
%                             end
%                             if distance_ring(end,1)==type
%                                 % correct classification
%                                 if type == 1 % SPU
%                                     true_pos_SPU(k) = true_pos_SPU(k) + 1;
%                                 elseif type == 2 % SPD
%                                     true_pos_SPD(k) = true_pos_SPD(k) + 1;
%                                 end
%                             else
%                                 % false classification 
%                                 if type == 1 % SPU would have been correct
%                                     if distance_ring(end,1)==2 % SPD was classified
%                                         false_pos_SPD(k) = false_pos_SPD(k) + 1;
%                                     end
%                                     false_neg_SPU(k) = false_neg_SPU(k) + 1;
% 
%                                 elseif type==2 % SPD would have been correct
%                                     if distance_ring(end,1)==1 % SPU was classified
%                                         false_pos_SPU(k) = false_pos_SPU(k) + 1;
%                                     end
%                                     false_neg_SPD(k) = false_neg_SPD(k) + 1;
% 
%                                 elseif distance_ring(end,1)==1 % SPU was classified and FIX/SHAUN would have been correct
%                                     false_pos_SPU(k) = false_pos_SPU(k) + 1;
% 
%                                 elseif distance_ring(end,1)==2 % SPD was classified and FIX/SHAUN would have been correct
%                                     false_pos_SPD(k) = false_pos_SPD(k) + 1;
%                                 end
%                             end
%                         end
%                         precision_SPU = true_pos_SPU ./(true_pos_SPU + false_pos_SPU);
%                         recall_SPU = true_pos_SPU ./(true_pos_SPU + false_neg_SPU);
%                         precision_SPD = true_pos_SPD ./(true_pos_SPD + false_pos_SPD);
%                         recall_SPD = true_pos_SPD ./(true_pos_SPD + false_neg_SPD);    
%                     end
%                 end
%                 % write results
%                 cnt = cnt + 1;
% 
%                 result(cnt,:)={subject vid sequencelength cutoff filter_coeff thresholds precision_SPU precision_SPD recall_SPU recall_SPD prior transmat mu Sigma mixmat mean_class_ma cov skip_samples_train skip_samples_test Perm config};
% 
%                 save('results.mat','result');
% 
%                 disp(['Subject: ' num2str(subject) ', Video:' num2str(vid) ' (Sequencelength: ' num2str(sequencelength) ', Cutoff: ' num2str(cutoff) ')']);
%                 for k = 1:round(length(thresholds)/5):length(thresholds)
%                     disp(['Threshold: ' num2str(thresholds(k))]);
%                     disp(['Pre_SPU: ' num2str(precision_SPU(k)) ', Pre_SPD: ' num2str(precision_SPD(k))]);
%                     disp(['Rec_SPU: ' num2str(recall_SPU(k)) ', Rec_SPD: ' num2str(recall_SPD(k))]);
%                 end
%                 diary('Log.txt')
% 
%             end
%         end
%     end
%     
%     % Train with all subjects
%     %% Generate Sequences
%     
%         disp(['Generate Sequences using all the data']);
% 
%         % Perm determines which sequences will be used for HMM training and
%         % which sequences will be used for generation of the classifier
%         [train_spu, val_spu, Perm{1}]=generate_random_ringpuffseq(spu_split_ord_subj,sequencelength,skip_samples_train,filter_coeff,g_h,h,d,cutoff);
%         [train_spd, val_spd, Perm{2}]=generate_random_ringpuffseq(spd_split_ord_subj,sequencelength,skip_samples_train,filter_coeff,g_h,h,d,cutoff);
%         [train_fix, val_fix, Perm{3}]=generate_random_ringpuffseq(fix_split_ord_subj,sequencelength,skip_samples_train,filter_coeff,g_h,h,d,cutoff);
%  
%     %% Train HMMs
%     
%         % initial guess of parameters
%         prior0 = normalise(rand(Q,1)); % prior(i) = Pr(Q(1) = i), --> Einsprungvektor
%         transmat0 = mk_stochastic(rand(Q,Q)); %transmat(i,j) = Pr(Q(t+1)=j | Q(t)=i) --> Transitionmatrix
%         %[mu0, Sigma0] = mixgauss_init(Q*M, train_spd{1}, cov_type);
%         [mu0, Sigma0] = mixgauss_init(Q*M, train_spd{1}, cov_type);
%         mu0 = reshape(mu0, [O Q M]);
%         Sigma0 = reshape(Sigma0, [O O Q M]);
%         %Sigma0= 0.4*ones(1,1,4);
%         mixmat0 = ones(4,1);
%         % Notation: Q(t) = hidden state, Y(t) = observation, M(t) = mixture variable
%         disp('Train HMMs');
%         disp(['SPU: ' num2str(length(train_spu)) '  SPD: ' num2str(length(train_spd)) '  FIX: ' num2str(length(train_fix)) ]);
% 
%         tic;
%         [LL, prior{1}, transmat{1}, mu{1}, Sigma{1}, mixmat{1}] = ...
%             mhmm_em(train_spu, prior0, transmat0, mu0, Sigma0, mixmat0,'max_iter',max_iter,'thresh',tol);
%         toc;
%         tic;
%         disp(sprintf('\tSPU-HMM finished'));
%         [~, prior{2}, transmat{2}, mu{2}, Sigma{2}, mixmat{2}] = ...
%             mhmm_em(train_spd, prior0, transmat0, mu0, Sigma0, mixmat0,'max_iter',max_iter,'thresh',tol);
%         toc;
%         tic;
%         disp(sprintf('\tSPD-HMM finished'));
%         [~, prior{3}, transmat{3}, mu{3}, Sigma{3}, mixmat{3}] = ...
%             mhmm_em(train_fix, prior0, transmat0, mu0, Sigma0, mixmat0,'max_iter',max_iter,'thresh',tol);
%         toc;
%         disp(sprintf('\tFIX-HMM finished'));
%     
%     %% Generate Mahalanobis Classifier
% 
%         disp('Generate Mahalanobis Classifier (FIX)');
%         [mean_class_ma{1}, ~ , cov{1}]=generate_classifier(val_spu, prior, transmat, mu, Sigma, mixmat,'ma');
%         disp('Generate Mahalanobis Classifier (SPU)');
%         [mean_class_ma{2}, ~ , cov{2}]=generate_classifier(val_spd, prior, transmat, mu, Sigma, mixmat,'ma');
%         disp('Generate Mahalanobis Classifier (SPD)');
%         [mean_class_ma{3}, ~ , cov{3}]=generate_classifier(val_fix, prior, transmat, mu, Sigma, mixmat,'ma');
%         
%     %% Save Data
%         cnt = cnt + 1;
%         result(cnt,:)={[] [] sequencelength cutoff filter_coeff [] [] [] [] [] prior transmat mu Sigma mixmat mean_class_ma cov  skip_samples_train [] Perm config};
%         save('results.mat','result');
%         diary('Log.txt')
% end
