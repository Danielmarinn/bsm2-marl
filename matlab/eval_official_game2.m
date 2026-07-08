%% eval_official_game2.m - Official BSM2 metrics for a saved Game2 trajectory
%
% Computes the official BSM2 plant-performance numbers (EQI, OCI and its
% components, 95th percentiles, effluent violations) over the standard
% evaluation window (days 245..609) from a trajectory .mat saved by
% RL_main_game2.m (save(..., '-v7.3') of the whole base workspace).
%
% This reuses the exact official computation validated in
% run_bsm2_manual_baseline.m (compute_official_bsm2_summary_from_base),
% with no plots, no prompts and no risk module, so it is safe in -batch.
%
% Usage (PowerShell, from the repository root):
%   & "C:\Program Files\MATLAB\R2025b\bin\matlab.exe" -batch "cd('matlab'); eval_official_game2"
%
% Optional: set TRAJFILE in the base workspace before calling to force a file,
% otherwise the bounded 609-day LAB trajectory below is used.

RL_DIR     = fileparts(fileparts(mfilename('fullpath')));
THESIS_DIR = fileparts(RL_DIR);
UNI_DIR    = fileparts(THESIS_DIR);
BSM2_DIR   = fullfile(THESIS_DIR, 'BSM2_R2019b');
if ~isfolder(BSM2_DIR)
    BSM2_DIR = fullfile(UNI_DIR, 'BSM2_R2019b');
end
if ~isfolder(BSM2_DIR)
    error('[eval] BSM2_R2019b folder not found. Checked LAB Thesis folder and Universidade folder.');
end
addpath(BSM2_DIR);
addpath(fullfile(RL_DIR, 'matlab'));

LOGS_DIR = fullfile(RL_DIR, 'logs');
OUT_DIR  = fullfile(RL_DIR, 'results');
DEFAULT_TRAJFILE = fullfile(LOGS_DIR, 'game2_traj_20260620_080319.mat');
CSVFILE = fullfile(LOGS_DIR, 'game2_sac_training_log.csv');

if exist('TRAJFILE', 'var') == 1 && ~isempty(TRAJFILE)
    trajfile = TRAJFILE;
else
    trajfile = DEFAULT_TRAJFILE;
    if exist(trajfile, 'file') ~= 2
        error('[eval] Trajectory file not found: %s', trajfile);
    end
end
fprintf('[eval] Trajectory file: %s\n', trajfile);

% Load the whole saved workspace into base (the helpers read via evalin base)
load(trajfile);
fprintf('[eval] Loaded. Variables in workspace: %d\n', numel(who));

% Quick diagnostic: are the energy inputs time-series (good) or scalars?
diag_vars = {'t','effluent','reac4','digesterout','kla4in','carbon1in'};
for k = 1:numel(diag_vars)
    nm = diag_vars{k};
    if exist(nm, 'var') == 1
        v = eval(nm);
        fprintf('[eval]   %-12s size [%s]  class %s\n', nm, num2str(size(v)), class(v));
    else
        fprintf('[eval]   %-12s MISSING\n', nm);
    end
end

% Training-log CSV: holds the time-aligned applied external carbon (Qec). The
% plant logs the carbon signal only coarsely in the .mat, so the carbon OCI
% term is recomputed from this CSV.
if exist(CSVFILE, 'file') ~= 2
    csvfile = '';
else
    csvfile = CSVFILE;
end

% Compute the official summary (same logic as the validated baseline pipeline)
summary = compute_official_bsm2_summary_from_base(NaN, csvfile);

% Write CSV
if ~exist(OUT_DIR, 'dir'), mkdir(OUT_DIR); end
stamp = datestr(now, 'yyyymmdd_HHMMSS');
outcsv = fullfile(OUT_DIR, ['game2_official_summary_' stamp '.csv']);
writetable(summary, outcsv);
fprintf('[eval] Wrote official summary CSV: %s\n', outcsv);

% Print the headline numbers
fprintf('\n===== OFFICIAL BSM2 METRICS (days 245..609) =====\n');
for r = 1:size(summary,1)
    fprintf('%-34s %12.4g\n', summary.metric{r}, summary.value(r));
end
fprintf('=================================================\n');
disp('EVAL_OFFICIAL_DONE');


%% ---- validated official computation (copied verbatim from run_bsm2_manual_baseline.m) ----

function summary = compute_official_bsm2_summary_from_base(elapsedSeconds, csvfile)
    if nargin < 2, csvfile = ''; end
    required = {'t','in','reac1','reac2','reac3','reac4','reac5','settler', ...
        'effluent','sludge','rec','primaryout','thickenerout','digesterin', ...
        'digesterout','dewateringout','storageout','qpassplant','qpassAS'};
    missing = {};
    for k = 1:numel(required)
        if evalin('base', sprintf('exist(''%s'', ''var'')', required{k})) ~= 1
            missing{end+1} = required{k}; %#ok<AGROW>
        end
    end

    if ~isempty(missing)
        summary = table( ...
            {'official_metrics_available'; 'elapsed_seconds'}, ...
            [0; elapsedSeconds], ...
            {'Missing required BSM2 workspace variables'; strjoin(missing, ', ')}, ...
            'VariableNames', {'metric','value','note'});
        return
    end

    t = evalin('base', 't');
    starttime = 245;
    stoptime = 609;
    startindex = max(find(t <= starttime)); %#ok<MXFND>
    stopindex = min(find(t >= stoptime)); %#ok<MXFND>
    time_eval = t(startindex:stopindex);
    timevector = time_eval(2:end) - time_eval(1:(end-1));
    totalt = time_eval(end) - time_eval(1);

    inpart = evalin('base', 'in');
    reac1part = evalin('base', 'reac1');
    reac2part = evalin('base', 'reac2');
    reac3part = evalin('base', 'reac3');
    reac4part = evalin('base', 'reac4');
    reac5part = evalin('base', 'reac5');
    settlerpart = evalin('base', 'settler');
    effluentpart = evalin('base', 'effluent');
    sludgepart = evalin('base', 'sludge');
    recpart = evalin('base', 'rec');
    primarypart = evalin('base', 'primaryout');
    thickenerpart = evalin('base', 'thickenerout');
    digesterinpart = evalin('base', 'digesterin');
    digesteroutpart = evalin('base', 'digesterout');
    dewateringoutpart = evalin('base', 'dewateringout');
    storagepart = evalin('base', 'storageout');
    qpassplantpart = evalin('base', 'qpassplant');
    qpassASpart = evalin('base', 'qpassAS');

    inpart = inpart(startindex:(stopindex-1), :);
    reac1part = reac1part(startindex:(stopindex-1), :);
    reac2part = reac2part(startindex:(stopindex-1), :);
    reac3part = reac3part(startindex:(stopindex-1), :);
    reac4part = reac4part(startindex:(stopindex-1), :);
    reac5part = reac5part(startindex:(stopindex-1), :);
    settlerpart = settlerpart(startindex:(stopindex-1), :);
    effluentpart = effluentpart(startindex:(stopindex-1), :);
    sludgepart = sludgepart(startindex:(stopindex-1), :);
    recpart = recpart(startindex:(stopindex-1), :);
    primarypart = primarypart(startindex:(stopindex-1), :);
    thickenerpart = thickenerpart(startindex:(stopindex-1), :);
    digesterinpart = digesterinpart(startindex:(stopindex-1), :);
    digesteroutpart = digesteroutpart(startindex:(stopindex-1), :);
    dewateringoutpart = dewateringoutpart(startindex:(stopindex-1), :);
    storagepart = storagepart(startindex:(stopindex-1), :);
    qpassplantpart = qpassplantpart(startindex:(stopindex-1), :);
    qpassASpart = qpassASpart(startindex:(stopindex-1), :);

    i_XB = bsm2_base_value('i_XB', 0.08);
    i_XP = bsm2_base_value('i_XP', 0.06);
    f_P = bsm2_base_value('f_P', 0.08);
    ACTIVATE = bsm2_base_value('ACTIVATE', 0);
    if ACTIVATE > 0.5
        warning('[eval] Dummy-variable ACTIVATE mode is not included in the compact official summary.');
    end

    BSS = 2;
    BCOD = 1;
    BNKj = 30;
    BNO = 10;
    BBOD5 = 2;

    totalCODemax = 100;
    totalNemax = 18;
    SNHemax = 4;
    TSSemax = 30;
    BOD5emax = 10;

    Qevec = effluentpart(:,15) .* timevector;
    Qetot = sum(Qevec);
    Qeav = Qetot / totalt;

    SSevec = effluentpart(:,2) .* Qevec;
    XIevec = effluentpart(:,3) .* Qevec;
    XSevec = effluentpart(:,4) .* Qevec;
    XBHevec = effluentpart(:,5) .* Qevec;
    XBAevec = effluentpart(:,6) .* Qevec;
    XPevec = effluentpart(:,7) .* Qevec;
    SNOevec = effluentpart(:,9) .* Qevec;
    SNHevec = effluentpart(:,10) .* Qevec;
    SNDevec = effluentpart(:,11) .* Qevec;
    XNDevec = effluentpart(:,12) .* Qevec;
    TSSevec = effluentpart(:,14) .* Qevec;

    qpassplantvec = qpassplantpart(:,15) .* timevector;
    qpassASvec = qpassASpart(:,15) .* timevector;
    SSbypassplantvec = qpassplantpart(:,2) .* qpassplantvec;
    XSbypassplantvec = qpassplantpart(:,4) .* qpassplantvec;
    XBHbypassplantvec = qpassplantpart(:,5) .* qpassplantvec;
    XBAbypassplantvec = qpassplantpart(:,6) .* qpassplantvec;
    SSbypassASvec = qpassASpart(:,2) .* qpassASvec;
    XSbypassASvec = qpassASpart(:,4) .* qpassASvec;
    XBHbypassASvec = qpassASpart(:,5) .* qpassASvec;
    XBAbypassASvec = qpassASpart(:,6) .* qpassASvec;

    totalNevec2 = (SNOevec + SNHevec + SNDevec + XNDevec + ...
        i_XB .* (XBHevec + XBAevec) + i_XP .* (XPevec + XIevec)) ./ Qevec;
    totalCODevec2 = (effluentpart(:,1).*Qevec + SSevec + XIevec + XSevec + ...
        XBHevec + XBAevec + XPevec) ./ Qevec;
    SNHevec2 = SNHevec ./ Qevec;
    TSSevec2 = TSSevec ./ Qevec;
    BOD5_SSloadvec = 0.25 .* (SSevec - SSbypassplantvec - SSbypassASvec) + ...
        0.65 .* (SSbypassplantvec + SSbypassASvec);
    BOD5_XSloadvec = 0.25 .* (XSevec - XSbypassplantvec - XSbypassASvec) + ...
        0.65 .* (XSbypassplantvec + XSbypassASvec);
    BOD5_XBHloadvec = 0.25 .* (1 - f_P) .* (XBHevec - XBHbypassplantvec - XBHbypassASvec) + ...
        0.65 .* (1 - f_P) .* (XBHbypassplantvec + XBHbypassASvec);
    BOD5_XBAloadvec = 0.25 .* (1 - f_P) .* (XBAevec - XBAbypassplantvec - XBAbypassASvec) + ...
        0.65 .* (1 - f_P) .* (XBAbypassplantvec + XBAbypassASvec);
    BOD5evec2 = (BOD5_SSloadvec + BOD5_XSloadvec + BOD5_XBHloadvec + BOD5_XBAloadvec) ./ Qevec;

    SSe = effluentpart(:,14);
    CODe = effluentpart(:,1) + effluentpart(:,2) + effluentpart(:,3) + ...
        effluentpart(:,4) + effluentpart(:,5) + effluentpart(:,6) + effluentpart(:,7);
    SNKje = effluentpart(:,10) + effluentpart(:,11) + effluentpart(:,12) + ...
        i_XB .* (effluentpart(:,5) + effluentpart(:,6)) + ...
        i_XP .* (effluentpart(:,3) + effluentpart(:,7));
    SNOe = effluentpart(:,9);
    EQIvec = (BSS .* SSe + BCOD .* CODe + BNKj .* SNKje + BNO .* SNOe + BBOD5 .* BOD5evec2) .* Qevec;
    EQI = sum(EQIvec) / (totalt * 1000);

    SSin = inpart(:,14);
    CODin = inpart(:,1) + inpart(:,2) + inpart(:,3) + inpart(:,4) + ...
        inpart(:,5) + inpart(:,6) + inpart(:,7);
    SNKjin = inpart(:,10) + inpart(:,11) + inpart(:,12) + ...
        i_XB .* (inpart(:,5) + inpart(:,6)) + i_XP .* (inpart(:,3) + inpart(:,7));
    SNOin = inpart(:,9);
    BOD5in = 0.65 .* (inpart(:,2) + inpart(:,4) + (1 - f_P) .* (inpart(:,5) + inpart(:,6)));
    Qinvec = inpart(:,15) .* timevector;
    IQIvec = (BSS .* SSin + BCOD .* CODin + BNKj .* SNKjin + BNO .* SNOin + BBOD5 .* BOD5in) .* Qinvec;
    IQI = sum(IQIvec) / (totalt * 1000);

    official_energy = compute_official_oci_components(timevector, totalt, startindex, stopindex, ...
        reac1part, reac2part, reac3part, reac4part, reac5part, settlerpart, sludgepart, ...
        primarypart, thickenerpart, digesterinpart, digesteroutpart, dewateringoutpart, ...
        storagepart, recpart);

    % Recompute the carbon OCI term from the time-aligned applied Qec (the .mat
    % logs the carbon signal coarsely, which biases the resampled value).
    CARBONSOURCECONC = bsm2_base_value('CARBONSOURCECONC', 400000);
    if ~isempty(csvfile) && exist(csvfile, 'file') == 2
        cmpd = carbon_from_csv(csvfile, CARBONSOURCECONC, starttime, stoptime);
        if isfinite(cmpd)
            official_energy.OCI = official_energy.OCI - official_energy.carbonmasscost + 3*cmpd;
            official_energy.carbonmasscost = 3 * cmpd;
            official_energy.carbonmassperd = cmpd;
        end
    end

    SNH95 = prctile(SNHevec2, 95);
    TN95 = prctile(totalNevec2, 95);
    TSS95 = prctile(TSSevec2, 95);

    [TN_v_time, TN_v_percent, TN_v_count] = violation_stats(totalNevec2 > totalNemax, timevector, totalt);
    [COD_v_time, COD_v_percent, COD_v_count] = violation_stats(totalCODevec2 > totalCODemax, timevector, totalt);
    [SNH_v_time, SNH_v_percent, SNH_v_count] = violation_stats(SNHevec2 > SNHemax, timevector, totalt);
    [TSS_v_time, TSS_v_percent, TSS_v_count] = violation_stats(TSSevec2 > TSSemax, timevector, totalt);
    [BOD_v_time, BOD_v_percent, BOD_v_count] = violation_stats(BOD5evec2 > BOD5emax, timevector, totalt);

    metric = {
        'official_metrics_available';
        'eval_start_day';
        'eval_stop_day';
        'eval_days';
        'elapsed_seconds';
        'IQI_kg_pollution_units_per_d';
        'EQI_kg_pollution_units_per_d';
        'OCI_total';
        'OCI_sludge_cost';
        'OCI_aeration_energy_cost';
        'OCI_pumping_energy_cost';
        'OCI_carbon_cost';
        'OCI_mixing_energy_cost';
        'OCI_heating_cost';
        'OCI_methane_credit';
        'airenergy_kWh_per_d';
        'pumpenergy_kWh_per_d';
        'mixenergy_kWh_per_d';
        'carbon_kgCOD_per_d';
        'methane_kgCH4_per_d';
        'effluent_flow_m3_per_d';
        'SNH95_gN_per_m3';
        'TN95_gN_per_m3';
        'TSS95_gSS_per_m3';
        'TN_violation_days';
        'TN_violation_percent';
        'TN_violation_count';
        'COD_violation_days';
        'COD_violation_percent';
        'COD_violation_count';
        'SNH_violation_days';
        'SNH_violation_percent';
        'SNH_violation_count';
        'TSS_violation_days';
        'TSS_violation_percent';
        'TSS_violation_count';
        'BOD5_violation_days';
        'BOD5_violation_percent';
        'BOD5_violation_count';
    };

    value = [
        1;
        starttime;
        stoptime;
        totalt;
        elapsedSeconds;
        IQI;
        EQI;
        official_energy.OCI;
        official_energy.TSScost;
        official_energy.airenergycost;
        official_energy.pumpenergycost;
        official_energy.carbonmasscost;
        official_energy.mixenergycost;
        official_energy.Heatenergycost;
        official_energy.EnergyfromMethaneperdcost;
        official_energy.airenergyperd;
        official_energy.pumpenergyperd;
        official_energy.mixenergyperd;
        official_energy.carbonmassperd;
        official_energy.Methaneprodperd;
        Qeav;
        SNH95;
        TN95;
        TSS95;
        TN_v_time;
        TN_v_percent;
        TN_v_count;
        COD_v_time;
        COD_v_percent;
        COD_v_count;
        SNH_v_time;
        SNH_v_percent;
        SNH_v_count;
        TSS_v_time;
        TSS_v_percent;
        TSS_v_count;
        BOD_v_time;
        BOD_v_percent;
        BOD_v_count;
    ];

    note = repmat({''}, numel(metric), 1);
    note{1} = 'Official-style metrics based on BSM2 perf_plant_bsm2.m definitions, excluding optional fuzzy risk prompt.';
    summary = table(metric, value, note, 'VariableNames', {'metric','value','note'});
end

function out = compute_official_oci_components(timevector, totalt, startindex, stopindex, ...
        reac1part, reac2part, reac3part, reac4part, reac5part, settlerpart, sludgepart, ...
        primarypart, thickenerpart, digesterinpart, digesteroutpart, dewateringoutpart, ...
        storagepart, recpart)

    VOL1 = bsm2_base_value('VOL1', NaN);
    VOL2 = bsm2_base_value('VOL2', NaN);
    VOL3 = bsm2_base_value('VOL3', NaN);
    VOL4 = bsm2_base_value('VOL4', NaN);
    VOL5 = bsm2_base_value('VOL5', NaN);
    DIM = bsm2_base_value('DIM', [NaN NaN]);
    VOL_P = bsm2_base_value('VOL_P', NaN);
    V_liq = bsm2_base_value('V_liq', NaN);
    CARBONSOURCECONC = bsm2_base_value('CARBONSOURCECONC', 400000);
    SOSAT1 = bsm2_base_value('SOSAT1', 8);
    SOSAT2 = bsm2_base_value('SOSAT2', 8);
    SOSAT3 = bsm2_base_value('SOSAT3', 8);
    SOSAT4 = bsm2_base_value('SOSAT4', 8);
    SOSAT5 = bsm2_base_value('SOSAT5', 8);
    P_atm = bsm2_base_value('P_atm', 1.013);
    R = bsm2_base_value('R', 0.083145);
    T_op = bsm2_base_value('T_op', 308.15);

    TSSreactors_start = (reac1part(1,14)*VOL1 + reac2part(1,14)*VOL2 + reac3part(1,14)*VOL3 + reac4part(1,14)*VOL4 + reac5part(1,14)*VOL5) / 1000;
    TSSreactors_end = (reac1part(end,14)*VOL1 + reac2part(end,14)*VOL2 + reac3part(end,14)*VOL3 + reac4part(end,14)*VOL4 + reac5part(end,14)*VOL5) / 1000;
    TSSsettler_start = sum(settlerpart(1,44:53)) * DIM(1) * DIM(2) / 10 / 1000;
    TSSsettler_end = sum(settlerpart(end,44:53)) * DIM(1) * DIM(2) / 10 / 1000;
    TSSprimary_start = primarypart(1,56) * VOL_P / 1000;
    TSSprimary_end = primarypart(end,56) * VOL_P / 1000;
    TSSdigester_start = digesteroutpart(1,14) * V_liq / 1000;
    TSSdigester_end = digesteroutpart(end,14) * V_liq / 1000;
    TSSstorage_start = storagepart(1,14) * storagepart(1,22) / 1000;
    TSSstorage_end = storagepart(end,14) * storagepart(end,22) / 1000;

    TSSsludgevec = (sludgepart(:,14) / 1000) .* sludgepart(:,15) .* timevector;
    TSSproduced = sum(TSSsludgevec) + TSSreactors_end + TSSsettler_end + TSSprimary_end + TSSdigester_end + TSSstorage_end - ...
        TSSreactors_start - TSSsettler_start - TSSprimary_start - TSSdigester_start - TSSstorage_start;
    TSSproducedperd = TSSproduced / totalt;

    n = evalin('base', 'size(t)');
    kla1in = vectorize_control_signal(bsm2_base_value('kla1in', 0), n);
    kla2in = vectorize_control_signal(bsm2_base_value('kla2in', 0), n);
    kla3in = vectorize_control_signal(bsm2_base_value('kla3in', 0), n);
    kla4in = vectorize_control_signal(bsm2_base_value('kla4in', 0), n);
    kla5in = vectorize_control_signal(bsm2_base_value('kla5in', 0), n);
    carbon1in = vectorize_control_signal(bsm2_base_value('carbon1in', 2), n);
    carbon2in = vectorize_control_signal(bsm2_base_value('carbon2in', 0), n);
    carbon3in = vectorize_control_signal(bsm2_base_value('carbon3in', 0), n);
    carbon4in = vectorize_control_signal(bsm2_base_value('carbon4in', 0), n);
    carbon5in = vectorize_control_signal(bsm2_base_value('carbon5in', 0), n);

    kla1vec = kla1in(startindex:(stopindex-1), :);
    kla2vec = kla2in(startindex:(stopindex-1), :);
    kla3vec = kla3in(startindex:(stopindex-1), :);
    kla4vec = kla4in(startindex:(stopindex-1), :);
    kla5vec = kla5in(startindex:(stopindex-1), :);
    airenergyvec = (SOSAT1*VOL1*kla1vec + SOSAT2*VOL2*kla2vec + SOSAT3*VOL3*kla3vec + SOSAT4*VOL4*kla4vec + SOSAT5*VOL5*kla5vec) / (1.8*1000);
    airenergyperd = sum(airenergyvec .* timevector) / totalt;

    mixenergyunitreac = 0.005;
    mixenergyreac = 24 * ( ...
        length(find(kla1vec < 20))*mixenergyunitreac*VOL1 + ...
        length(find(kla2vec < 20))*mixenergyunitreac*VOL2 + ...
        length(find(kla3vec < 20))*mixenergyunitreac*VOL3 + ...
        length(find(kla4vec < 20))*mixenergyunitreac*VOL4 + ...
        length(find(kla5vec < 20))*mixenergyunitreac*VOL5) * (timevector(1));
    mixenergyAD = 24 * 0.005 * V_liq * totalt;
    mixenergyperd = (mixenergyreac + mixenergyAD) / totalt;

    PF_Qintr = 0.004;
    PF_Qr = 0.008;
    PF_Qw = 0.05;
    PF_Qpu = 0.075;
    PF_Qtu = 0.060;
    PF_Qdo = 0.004;
    pumpenergyvec = PF_Qintr*recpart(:,15) + PF_Qr*settlerpart(:,15) + PF_Qw*settlerpart(:,22) + ...
        PF_Qpu*primarypart(:,36) + PF_Qtu*thickenerpart(:,15) + PF_Qdo*dewateringoutpart(:,36);
    pumpenergyperd = sum(pumpenergyvec .* timevector) / totalt;

    carbon1vec = carbon1in(startindex:(stopindex-1), :);
    carbon2vec = carbon2in(startindex:(stopindex-1), :);
    carbon3vec = carbon3in(startindex:(stopindex-1), :);
    carbon4vec = carbon4in(startindex:(stopindex-1), :);
    carbon5vec = carbon5in(startindex:(stopindex-1), :);
    carbonmassvec = (carbon1vec + carbon2vec + carbon3vec + carbon4vec + carbon5vec) * CARBONSOURCECONC / 1000;
    carbonmassperd = sum(carbonmassvec .* timevector) / totalt;

    Methanevec = digesteroutpart(:,48) ./ digesteroutpart(:,50) * P_atm * 16 / (R*T_op);
    Methaneflowvec = Methanevec .* digesteroutpart(:,51);
    Methaneprodperd = sum(Methaneflowvec .* timevector) / totalt;

    Tdigesterin = (primarypart(:,37).*primarypart(:,36) + thickenerpart(:,16).*thickenerpart(:,15)) ./ ...
        (primarypart(:,36) + thickenerpart(:,15));
    Heatpower = (35 - Tdigesterin) .* digesterinpart(:,27) * 1000 * 4.186 / 86400;
    Heatenergyperd = 24 * sum(Heatpower .* timevector) / totalt;

    out.TSScost = 3 * TSSproducedperd;
    out.airenergycost = airenergyperd;
    out.mixenergycost = mixenergyperd;
    out.pumpenergycost = pumpenergyperd;
    out.carbonmasscost = 3 * carbonmassperd;
    out.EnergyfromMethaneperdcost = 6 * Methaneprodperd;
    out.Heatenergycost = max(0, Heatenergyperd - 7*Methaneprodperd);
    out.OCI = out.TSScost + out.airenergycost + out.mixenergycost + out.pumpenergycost + ...
        out.carbonmasscost - out.EnergyfromMethaneperdcost + out.Heatenergycost;
    out.airenergyperd = airenergyperd;
    out.pumpenergyperd = pumpenergyperd;
    out.mixenergyperd = mixenergyperd;
    out.carbonmassperd = carbonmassperd;
    out.Methaneprodperd = Methaneprodperd;
end

function cmpd = carbon_from_csv(csvfile, CONC, a, b)
    % Time-weighted mean applied external carbon (Qec) over [a,b) -> kg COD/d.
    cmpd = NaN;
    try
        opts = detectImportOptions(csvfile);
        qcol = 'Qec_applied_for_reward';
        if ~ismember(qcol, opts.VariableNames), qcol = 'Qec'; end
        opts.SelectedVariableNames = {'time', qcol};
        T = readtable(csvfile, opts);
    catch
        return
    end
    tt = T.time;
    q = T.(qcol);
    ok = isfinite(tt) & isfinite(q);
    tt = tt(ok); q = q(ok);
    [tt, ix] = sort(tt); q = q(ix);
    if numel(tt) < 2, return; end
    dt = [diff(tt); 0];
    dt = min(max(dt, 0), 0.05);
    m = tt >= a & tt < b;
    if ~any(m), return; end
    cmpd = sum(q(m) .* dt(m)) / sum(dt(m)) * CONC / 1000;
end

function value = bsm2_base_value(name, defaultValue)
    if evalin('base', sprintf('exist(''%s'', ''var'')', name)) == 1
        value = evalin('base', name);
    else
        value = defaultValue;
    end
end

function out = vectorize_control_signal(invariable, outvecsize)
    N = outvecsize(1);
    m = size(invariable, 1);
    if m <= 1
        out = ones(outvecsize) .* invariable;          % scalar -> constant vector
    elseif m == N
        out = invariable;                               % already aligned to t
    else
        % Control input logged at a coarser/different rate than t (e.g. the
        % external-carbon signal). Zero-order-hold resample onto the t grid,
        % assuming the signal spans the full simulation uniformly.
        src = linspace(0, 1, m)';
        dst = linspace(0, 1, N)';
        out = interp1(src, invariable, dst, 'previous', 'extrap');
    end
end

function [days, percent, count] = violation_stats(mask, timevector, totalt)
    idx = find(mask);
    days = min(totalt, numel(idx) * timevector(1));
    percent = min(100, days / totalt * 100);
    if isempty(idx)
        count = 0;
        return
    end
    count = 1 + sum(diff(idx) > 1);
end
