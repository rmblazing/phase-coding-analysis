
PCx_files{1} ='Z:\robin\Shiva_data\Nagappan2021_data\eLife2021_DryadData\eLife2021_DryadData\Ntng\190911_2\190911-001_bank1.efd';
PCx_files{2} ='Z:\robin\Shiva_data\Nagappan2021_data\eLife2021_DryadData\eLife2021_DryadData\Ntng\190912\190912-001_bank1.efd';
PCx_files{3} ='Z:\robin\Shiva_data\Nagappan2021_data\eLife2021_DryadData\eLife2021_DryadData\Ntng\190913\190913-001_bank1.efd';
PCx_files{4} ='Z:\robin\Shiva_data\Nagappan2021_data\eLife2021_DryadData\eLife2021_DryadData\Ntng\191121\191121-001_bank1.efd';
PCx_files{5} ='Z:\robin\Shiva_data\Nagappan2021_data\eLife2021_DryadData\eLife2021_DryadData\Ntng\191121_2\191121-005_bank1.efd';
PCx_files{6} ='Z:\robin\Shiva_data\Nagappan2021_data\eLife2021_DryadData\eLife2021_DryadData\Ntng\200624\200624-001_bank1.efd';
PCx_files{7} ='Z:\robin\Shiva_data\Nagappan2021_data\eLife2021_DryadData\eLife2021_DryadData\Ntng\201002\201002-002_bank1.efd';
PCx_files{8} ='Z:\robin\Shiva_data\Nagappan2021_data\eLife2021_DryadData\eLife2021_DryadData\Ntng\201006_2\201006-003_bank1.efd';
PCx_files{9} ='Z:\robin\Shiva_data\Nagappan2021_data\eLife2021_DryadData\eLife2021_DryadData\Ntng\201011\201011-001_bank1.efd';
PCx_files{10} ='Z:\robin\Shiva_data\Nagappan2021_data\eLife2021_DryadData\eLife2021_DryadData\Ntng\201013\201013-001_bank1.efd';
PCx_files{11} ='Z:\robin\Shiva_data\Nagappan2021_data\eLife2021_DryadData\eLife2021_DryadData\Ntng\201015\201015-001_bank1.efd';
PCx_files{12} ='Z:\robin\Shiva_data\Nagappan2021_data\eLife2021_DryadData\eLife2021_DryadData\Ntng\201024\201024-001_bank1.efd';
PCx_files{13} ='Z:\robin\Shiva_data\Nagappan2021_data\eLife2021_DryadData\eLife2021_DryadData\Ntng\201028\201028-001_bank1.efd';
PCx_files{14} ='Z:\robin\Shiva_data\Nagappan2021_data\eLife2021_DryadData\eLife2021_DryadData\Ntng\201124\201124-002_bank1.efd';
PCx_files{15} ='Z:\robin\Shiva_data\Nagappan2021_data\eLife2021_DryadData\eLife2021_DryadData\Ntng\201205\201205-001_bank1.efd';
PCx_files{16} ='Z:\robin\Shiva_data\Nagappan2021_data\eLife2021_DryadData\eLife2021_DryadData\Ntng\210103\210103-001_bank1.efd';
PCx_files{17} ='Z:\robin\Shiva_data\Nagappan2021_data\eLife2021_DryadData\eLife2021_DryadData\Ntng\210129\210129-001_bank1.efd';


for k = 1:length(PCx_files)
    PCx_efd = load(PCx_files{k},'-mat');
    RasterAlign = PCx_efd.efd.ValveSpikes.RasterAlign;    
    n_cells = size(RasterAlign);
    allncells(k) = n_cells(3);
    date_ID = regexp(PCx_files{k},'\','split');
    savepath = [fileparts(PCx_files{k}), '\', date_ID{7}, '_RasterAlign.mat'];
    save(savepath, 'RasterAlign');
end

for k = 1:length(PCx_files)
    PCx_efd = load(PCx_files{k},'-mat');
    FVSwitchTimes = PCx_efd.efd.ValveTimes.FVSwitchTimesOn;    
    date_ID = regexp(PCx_files{k},'\','split');
    savepath = [fileparts(PCx_files{k}), '\', date_ID{7}, '_FVSwitchTimes.mat'];
    save(savepath, 'FVSwitchTimes');
end

for k = 1:length(PCx_files)
    PCx_efd = load(PCx_files{k},'-mat');
    PREXTimes = PCx_efd.efd.ValveTimes.PREXTimes;    
    date_ID = regexp(PCx_files{k},'\','split');
    savepath = [fileparts(PCx_files{k}), '\', date_ID{7}, '_PREXtimes.mat'];
    save(savepath, 'PREXTimes');
end


resp_files{1} ='Z:\robin\Shiva_data\Nagappan2021_data\eLife2021_DryadData\eLife2021_DryadData\Ntng\190911_2\190911-001.resp';
resp_files{2} ='Z:\robin\Shiva_data\Nagappan2021_data\eLife2021_DryadData\eLife2021_DryadData\Ntng\190912\190912-001.resp';
resp_files{3} ='Z:\robin\Shiva_data\Nagappan2021_data\eLife2021_DryadData\eLife2021_DryadData\Ntng\190913\190913-001.resp';
resp_files{4} ='Z:\robin\Shiva_data\Nagappan2021_data\eLife2021_DryadData\eLife2021_DryadData\Ntng\191121\191121-001.resp';
resp_files{5} ='Z:\robin\Shiva_data\Nagappan2021_data\eLife2021_DryadData\eLife2021_DryadData\Ntng\191121_2\191121-005.resp';
resp_files{6} ='Z:\robin\Shiva_data\Nagappan2021_data\eLife2021_DryadData\eLife2021_DryadData\Ntng\200624\200624-001.resp';
resp_files{7} ='Z:\robin\Shiva_data\Nagappan2021_data\eLife2021_DryadData\eLife2021_DryadData\Ntng\201002\201002-002.resp';
resp_files{8} ='Z:\robin\Shiva_data\Nagappan2021_data\eLife2021_DryadData\eLife2021_DryadData\Ntng\201006_2\201006-003.resp';
resp_files{9} ='Z:\robin\Shiva_data\Nagappan2021_data\eLife2021_DryadData\eLife2021_DryadData\Ntng\201011\201011-001.resp';
resp_files{10} ='Z:\robin\Shiva_data\Nagappan2021_data\eLife2021_DryadData\eLife2021_DryadData\Ntng\201013\201013-001.resp';
resp_files{11} ='Z:\robin\Shiva_data\Nagappan2021_data\eLife2021_DryadData\eLife2021_DryadData\Ntng\201015\201015-001.resp';
resp_files{12} ='Z:\robin\Shiva_data\Nagappan2021_data\eLife2021_DryadData\eLife2021_DryadData\Ntng\201024\201024-001.resp';
resp_files{13} ='Z:\robin\Shiva_data\Nagappan2021_data\eLife2021_DryadData\eLife2021_DryadData\Ntng\201028\201028-001.resp';
resp_files{14} ='Z:\robin\Shiva_data\Nagappan2021_data\eLife2021_DryadData\eLife2021_DryadData\Ntng\201124\201124-002.resp';
resp_files{15} ='Z:\robin\Shiva_data\Nagappan2021_data\eLife2021_DryadData\eLife2021_DryadData\Ntng\201205\201205-001.resp';
resp_files{16} ='Z:\robin\Shiva_data\Nagappan2021_data\eLife2021_DryadData\eLife2021_DryadData\Ntng\210103\210103-001.resp';
resp_files{17} ='Z:\robin\Shiva_data\Nagappan2021_data\eLife2021_DryadData\eLife2021_DryadData\Ntng\210129\210129-001.resp';


for k = 1:length(resp_files)
    resp = load(resp_files{k},'-mat');
    resp_array = resp.RRR;
    date_ID = regexp(PCx_files{k},'\','split');
    savepath = [fileparts(PCx_files{k}), '\', date_ID{7}, '_resp.mat'];
    save(savepath, 'resp_array');
    clear resp
end

%% Get indices of SLs and PYRs
ROI = {'SL'};
Catalog = 'Z:\robin\Shiva_data\Nagappan2021_data\eLife2021_DryadData\eLife2021_DryadData\ExperimentCatalog_Ntng.txt'; % set to data and catalog directory
T = readtable(Catalog, 'Delimiter', ' ');
ROIfiles = T.kwikfile(logical(T.include) & strcmp(T.ROI,ROI));

LRcells = LRcellPicker_chgPt(ROIfiles{1},[-.1 .1]);
LR_idx_SL = LRcells.primLR;
LR_idx_Pyr = LRcells.nonLR;

for file = 2:length(ROIfiles)
    LRcells = LRcellPicker_chgPt(ROIfiles{file},[-.1 .1]);
    all_cells = cat(2,LR_idx_SL, LR_idx_Pyr);
    max_idx = max(all_cells);
    LR_idx_SL = cat(2,LR_idx_SL, LRcells.primLR + max_idx);
    LR_idx_Pyr  = cat(2,LR_idx_Pyr, LRcells.nonLR + max_idx);
end

 all_cells = cat(2,LR_idx_SL, LR_idx_Pyr);

 %% get indices of PYRs and SLs for each experiment 
for file = 1:length(ROIfiles)
    LRcells = LRcellPicker_chgPt(ROIfiles{file},[-.1 .1]);
    LR_idx_SL_track{file} = LRcells.primLR;
    LR_idx_Pyr_track{file}  = LRcells.nonLR;
end

savepath = 'Z:\robin\Shiva_data\SL_indices'
save(savepath, 'LR_idx_SL');
savepath = 'Z:\robin\Shiva_data\Pyr_indices'
save(savepath, 'LR_idx_Pyr');
savepath = 'Z:\robin\Shiva_data\SL_indices_indiv_experiments'
save(savepath, 'LR_idx_SL_track');
savepath = 'Z:\robin\Shiva_data\Pyr_indices_indiv_experiments'
save(savepath, 'LR_idx_Pyr_track');

load('S:\All_Staff\robin\Shiva_data\Nagappan2021_data\eLife2021_DryadData\eLife2021_DryadData\Ntng\191121_2\191121-005_bank1.efd','-mat')
Trials = [1:15]
for T = 1:length(Trials)
 RA(T).Times = efd.ValveSpikes.RasterAlign{1,6}{Trials(T)};
end
PST = [-.5 1.5];
KernelSize = 0.01;
[KDF, KDFt, KDFe] = psth(RA,KernelSize,'n',PST);