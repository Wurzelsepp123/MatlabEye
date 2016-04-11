load('results.mat');
close all

thresholds=result{2,6};

sum=0;
for j=2:19
    sum= sum + result{j,7};
end
pre_spu_mean=sum/18;
sum=0;
for j=2:19
    sum= sum + result{j,8};
end
pre_spd_mean=sum/18;
sum=0;
for j=2:19
    sum= sum + result{j,9};
end
rec_spu_mean=sum/18;
sum=0;
for j=2:19
    sum= sum + result{j,10};
end
rec_spd_mean=sum/18;


sum=0;
for j=25:42
    sum= sum + result{j,7};
end
pre_spu_mean2=sum/18;
sum=0;
for j=25:42
    sum= sum + result{j,8};
end
pre_spd_mean2=sum/18;
sum=0;
for j=2:19
    sum= sum + result{j,9};
end
rec_spu_mean2=sum/18;
sum=0;
for j=25:42
    sum= sum + result{j,10};
end
rec_spd_mean2=sum/18;


sum=0;
for j=44:61
    sum= sum + result{j,7};
end
pre_spu_mean250=sum/18;
sum=0;
for j=44:61
    sum= sum + result{j,8};
end
pre_spd_mean250=sum/18;
sum=0;
for j=44:61
    sum= sum + result{j,9};
end
rec_spu_mean250=sum/18;
sum=0;
for j=44:61
    sum= sum + result{j,10};
end
rec_spd_mean250=sum/18;


%% SPU
l=1;
subplot(1,2,1)
plot(pre_spu_mean(1:l:end),rec_spu_mean(1:l:end),'-o')
hold on
plot(pre_spu_mean2(1:l:end),rec_spu_mean2(1:l:end),'-o')
plot(pre_spu_mean250(1:l:end),rec_spu_mean250(1:l:end),'-o')

% Optimal Threshold
idx_max=find(max(pre_spu_mean+rec_spu_mean)==(pre_spu_mean+rec_spu_mean));
plot(pre_spu_mean(idx_max),rec_spu_mean(idx_max),'LineStyle','none','Marker','x','MarkerSize',15,'LineWidth',2)
text(pre_spu_mean(idx_max),rec_spu_mean(idx_max),num2str(thresholds(idx_max)))

idx_max=find(max(pre_spu_mean250+rec_spu_mean250)==(pre_spu_mean250+rec_spu_mean250));
plot(pre_spu_mean250(idx_max),rec_spu_mean250(idx_max),'LineStyle','none','Marker','x','MarkerSize',15,'LineWidth',2)
text(pre_spu_mean250(idx_max),rec_spu_mean250(idx_max),num2str(thresholds(idx_max)))

% Label
xlabel('Precision');
ylabel('Recall');
legend('Sequenzlength = 300; Cutoff=1','Sequenzlength = 300; Cutoff=0','Sequenzlength = 250; Cutoff=1','Optimal Threshold; Sequenzlength = 300; Cutoff=1','Optimal Threshold; Sequenzlength = 250; Cutoff=1','location','west');
title('Recall vs Precision (mean of all subjects) - SPU');
ylim([0.5 1])
grid on
for k=1:5*l:length(pre_spu_mean)
  %  plot([pre_spu_mean(k) pre_spd_mean(k)],[rec_spu_mean(k) rec_spd_mean(k)],'LineStyle','--','Color','black','LineWidth',0.5)
    text(pre_spu_mean(k),rec_spu_mean(k),num2str(thresholds(k)))
    text(pre_spu_mean2(k),rec_spu_mean2(k),num2str(thresholds(k)))
    text(pre_spu_mean250(k),rec_spu_mean250(k),num2str(thresholds(k)))
end

%% SPD
subplot(1,2,2)
plot(pre_spd_mean(1:l:end),rec_spd_mean(1:l:end),'-o')
hold on
plot(pre_spd_mean2(1:l:end),rec_spd_mean2(1:l:end),'-o')
plot(pre_spd_mean250(1:l:end),rec_spd_mean250(1:l:end),'-o')


% Optimal Threshold
idx_max=find(max(pre_spd_mean+rec_spd_mean)==(pre_spd_mean+rec_spd_mean));
plot(pre_spd_mean(idx_max),rec_spd_mean(idx_max),'LineStyle','none','Marker','x','MarkerSize',15,'LineWidth',2)
text(pre_spd_mean(idx_max),rec_spd_mean(idx_max),num2str(thresholds(idx_max)))

idx_max=find(max(pre_spd_mean250+rec_spd_mean250)==(pre_spd_mean250+rec_spd_mean250));
plot(pre_spd_mean250(idx_max),rec_spd_mean250(idx_max),'LineStyle','none','Marker','x','MarkerSize',15,'LineWidth',2)
text(pre_spd_mean250(idx_max),rec_spd_mean250(idx_max),num2str(thresholds(idx_max)))

% Label
xlabel('Precision');
ylabel('Recall');
legend('Sequenzlength = 300; Cutoff=1','Sequenzlength = 300; Cutoff=0','Sequenzlength = 250; Cutoff=1','Optimal Threshold; Sequenzlength = 300; Cutoff=1','Optimal Threshold; Sequenzlength = 250; Cutoff=1','location','west');
title('Recall vs Precision (mean of all subjects) - SPD');
ylim([0.5 1])
grid on
for k=1:5*l:length(pre_spu_mean)
  %  plot([pre_spu_mean(k) pre_spd_mean(k)],[rec_spu_mean(k) rec_spd_mean(k)],'LineStyle','--','Color','black','LineWidth',0.5)
    text(pre_spd_mean(k),rec_spd_mean(k),num2str(thresholds(k)))
    text(pre_spd_mean2(k),rec_spd_mean2(k),num2str(thresholds(k)))   
    text(pre_spd_mean250(k),rec_spd_mean250(k),num2str(thresholds(k)))
end


%% Optimal threshold:

idx_max = find(max(pre_spu_mean+rec_spu_mean)==(pre_spu_mean+rec_spu_mean));
disp(['Seq = 300: Max(Pre_SPU (' num2str(pre_spu_mean(idx_max)) ') + Rec_SPU (' num2str(rec_spu_mean(idx_max)) ')) at threshold = ' num2str(result{2,6}(idx_max))]);
idx_max = find(max(pre_spd_mean+rec_spd_mean)==(pre_spd_mean+rec_spd_mean));
disp(['Seq = 300: Max(Pre_SPD (' num2str(pre_spd_mean(idx_max)) ') + Rec_SPD (' num2str(rec_spd_mean(idx_max)) ')) at threshold = ' num2str(result{2,6}(idx_max))]);


idx_max = find(max(pre_spu_mean250+rec_spu_mean250)==(pre_spu_mean250+rec_spu_mean250));
disp(['Seq = 250: Max(Pre_SPU (' num2str(pre_spu_mean250(idx_max)) ') + Rec_SPU (' num2str(rec_spu_mean250(idx_max)) ')) at threshold = ' num2str(result{2,6}(idx_max))]);
idx_max = find(max(pre_spd_mean250+rec_spd_mean250)==(pre_spd_mean250+rec_spd_mean250));
disp(['Seq = 250: Max(Pre_SPD (' num2str(pre_spd_mean250(idx_max)) ') + Rec_SPD (' num2str(rec_spd_mean250(idx_max)) ')) at threshold = ' num2str(result{2,6}(idx_max))]);

