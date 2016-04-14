load('results.mat');
close all

start_row_HMM = 2;
start_row_SVM = 21;


thresholds=result{2,6};
%% HMM
sum=0;
for j=start_row_HMM:start_row_HMM+17
    sum= sum + result{j,7};
end
pre_spu_mean=sum/18;
sum=0;
for j=start_row_HMM:start_row_HMM+17
    sum= sum + result{j,8};
end
pre_spd_mean=sum/18;
sum=0;
for j=start_row_HMM:start_row_HMM+17
    sum= sum + result{j,9};
end
rec_spu_mean=sum/18;
sum=0;
for j=start_row_HMM:start_row_HMM+17
    sum= sum + result{j,10};
end
rec_spd_mean=sum/18;

pre_mean=(pre_spd_mean+pre_spu_mean)/2;
rec_mean=(rec_spd_mean+rec_spu_mean)/2;


%% SVM

sum=0;
for j=start_row_SVM:start_row_SVM+17
    sum= sum + result{j,7};
end
pre_spu_mean_SVM=sum/18;
sum=0;
for j=start_row_SVM:start_row_SVM+17
    sum= sum + result{j,8};
end
pre_spd_mean_SVM=sum/18;
sum=0;
for j=start_row_SVM:start_row_SVM+17
    sum= sum + result{j,9};
end
rec_spu_mean_SVM=sum/18;
sum=0;
for j=start_row_SVM:start_row_SVM+17
    sum= sum + result{j,10};
end
rec_spd_mean_SVM=sum/18;

pre_mean_SVM =(pre_spd_mean_SVM+pre_spu_mean_SVM)/2;
rec_mean_SVM =(rec_spd_mean_SVM+rec_spu_mean_SVM)/2;

%% Plot
l=1;
plot(pre_spu_mean(1:l:end),rec_spu_mean(1:l:end),'-o','LineWidth',2,'MarkerSize',8)
hold on
plot(pre_spd_mean(1:l:end),rec_spd_mean(1:l:end),'-o','LineWidth',2,'MarkerSize',8)
%plot(pre_mean_SVM(1:l:end),rec_mean_SVM(1:l:end),'-o')

% Optimal Threshold
idx_max=find(max(pre_spu_mean+rec_spu_mean)==(pre_spu_mean+rec_spu_mean));
plot(pre_spu_mean(idx_max),rec_spu_mean(idx_max),'LineStyle','none','Marker','x','MarkerSize',30,'LineWidth',3)
text(pre_spu_mean(idx_max),rec_spu_mean(idx_max),num2str(thresholds(idx_max)))

idx_max=find(max(pre_spd_mean+rec_spd_mean)==(pre_spd_mean+rec_spd_mean));
plot(pre_spd_mean(idx_max),rec_spd_mean(idx_max),'LineStyle','none','Marker','x','MarkerSize',30,'LineWidth',3)
text(pre_spd_mean(idx_max),rec_spd_mean(idx_max),num2str(thresholds(idx_max)))

% idx_max=find(max(pre_mean_SVM+rec_mean_SVM)==(pre_mean_SVM+rec_mean_SVM));
% plot(pre_mean_SVM(idx_max),rec_mean_SVM(idx_max),'LineStyle','none','Marker','x','MarkerSize',15,'LineWidth',2)
% text(pre_mean_SVM(idx_max),rec_mean_SVM(idx_max),num2str(thresholds(idx_max)))

% Label
xlabel('Precision');
ylabel('Recall');
restoredefaultpath
rehash toolboxcache
%legend('HMM','SVM','HMM optimal Threshold','SVM optimal threshold','location','west');
legend('HMM SPU','HMM SPD','HMM SPU optimal Threshold','HMM SPD optimal Threshold','location','west');
title('Recall vs Precision (mean of all subjects) - sequencelength 300');
ylim([0 1])
grid on
for k=1:5*l:length(pre_mean)
    text(pre_spu_mean(k),rec_spu_mean(k),num2str(thresholds(k)),'FontSize',16,'FontWeight','bold')
    text(pre_spd_mean(k),rec_spd_mean(k),num2str(thresholds(k)),'FontSize',16,'FontWeight','bold')
  %  text(pre_mean_SVM(k),rec_mean_SVM(k),num2str(thresholds(k)))
end


%% Optimal threshold:

idx_max = find(max(pre_spu_mean+rec_spu_mean)==(pre_spu_mean+rec_spu_mean));
disp(['Seq = 300: Max(Pre_SPU (' num2str(pre_spu_mean(idx_max)) ') + Rec_SPU (' num2str(rec_spu_mean(idx_max)) ')) at threshold = ' num2str(result{2,6}(idx_max))]);
idx_max = find(max(pre_spd_mean+rec_spd_mean)==(pre_spd_mean+rec_spd_mean));
disp(['Seq = 300: Max(Pre_SPD (' num2str(pre_spd_mean(idx_max)) ') + Rec_SPD (' num2str(rec_spd_mean(idx_max)) ')) at threshold = ' num2str(result{2,6}(idx_max))]);

idx_max = find(max(pre_mean+rec_mean)==(pre_mean+rec_mean));
disp(['Seq = 300: Max(Pre_Mean(SPD/SPU) (' num2str(pre_mean(idx_max)) ') + Rec_SPD (' num2str(rec_mean(idx_max)) ')) at threshold = ' num2str(result{2,6}(idx_max))]);


idx_max = find(max(pre_mean_SVM+rec_mean_SVM)==(pre_mean_SVM+rec_mean_SVM));
disp(['Seq = 300: Max(Pre_Mean(SPD/SPU) (' num2str(pre_mean_SVM(idx_max)) ') + Rec_SPD (' num2str(rec_mean_SVM(idx_max)) ')) at threshold = ' num2str(result{2,6}(idx_max))]);




