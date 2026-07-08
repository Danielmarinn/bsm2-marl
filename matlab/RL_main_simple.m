%% matlab/RL_main_simple.m - Orquestrador simples (t=0 a t=609)
%
% Usado para treinar UM agente de cada vez. A variavel AGENT no topo
% escolhe qual variavel manipulada e' controlada pelo Python:
%   'qint' -> CTRL-2 (recirculacao interna). Qec fica no default BSM2 (2 m3/d).
%   'qec'  -> CTRL-1 (carbono externo no R1). Qint fica no default BSM2 (61944 m3/d).
%   'do'   -> CTRL-3 (SO4ref no loop oficial DO do R4). Qec e Qint ficam nos defaults BSM2.
%   'qw'   -> CTRL-4 (waste sludge / Qw). Decide em janela diaria; os outros
%             controlos ficam nos defaults BSM2.
%
% Os tres agentes sao mutuamente independentes: cada um tem o seu
% proprio checkpoint e o seu proprio log. Nao existe qualquer partilha
% de estado entre runs - alternar entre agentes e' apenas editar a linha
% AGENT abaixo.
%
% IMPORTANTE: a simulacao corre desde t=0 (nao desde t=245). Arrancar
% em t=245 em accelerator mode completa instantaneamente sem dar tempo
% ao loop para pausar. Os primeiros 245 dias sao estabilizacao - os
% agentes SAC so aprendem efectivamente a partir do dia 245.
%
% ANTES DE CORRER:
%   1. Em MATLAB: init_bsm2
%   2. Verificar que AGENT abaixo esta no valor desejado.
%   3. No terminal, arrancar o agente correspondente:
%        AGENT='qint' -> python agents/ctrl_sac_qint.py
%        AGENT='qec'  -> python agents/ctrl_sac_qec.py
%        AGENT='do'   -> python agents/ctrl_sac_do.py
%        AGENT='qw'   -> python agents/ctrl_sac_qw.py
%   4. Em MATLAB: run RL_main_simple
%   (NAO carregar states_day245.mat)

%% ===============================
% Paths do projecto
%% ===============================
RL_DIR   = fileparts(fileparts(mfilename('fullpath')));
THESIS_DIR = fileparts(RL_DIR);
UNI_DIR    = fileparts(THESIS_DIR);
BSM2_DIR   = fullfile(THESIS_DIR, 'BSM2_R2019b');
if ~isfolder(BSM2_DIR)
    BSM2_DIR = fullfile(UNI_DIR, 'BSM2_R2019b');
end
if ~isfolder(BSM2_DIR)
    error('[RL_main] BSM2_R2019b folder not found. Checked Thesis work and Universidade folders.');
end

cd(BSM2_DIR);
addpath(BSM2_DIR);

model    = 'bsm2_cl';
COMMS    = fullfile(RL_DIR, 'comms');
addpath(fullfile(RL_DIR, 'matlab'));

if bdIsLoaded(model)
    try
        simStatus = get_param(model, 'SimulationStatus');
    catch
        simStatus = 'stopped';
    end

    if ~strcmpi(simStatus, 'stopped')
        fprintf('[RL_main] Modelo %s ainda estava %s. A parar antes de recarregar...\n', ...
            model, simStatus);
        try
            set_param(model, 'SimulationCommand', 'stop');
        catch stopErr
            warning('[RL_main] Falhou o pedido de stop ao modelo: %s', stopErr.message);
        end

        tStop = tic;
        while bdIsLoaded(model)
            try
                simStatus = get_param(model, 'SimulationStatus');
            catch
                simStatus = 'stopped';
            end

            if strcmpi(simStatus, 'stopped')
                break;
            end

            if toc(tStop) > 60
                error('[RL_main] Timeout a aguardar que %s parasse antes de recarregar.', model);
            end
            pause(0.2);
        end
    end

    try
        close_system(model, 0);
    catch closeErr
        warning('[RL_main] close_system falhou na primeira tentativa: %s', closeErr.message);
        pause(0.5);
        close_system(model, 0);
    end
end
load_system(model);

%% ===============================
% Seleccao do agente activo
%   'qint' -> treina CTRL-2, update_Qint_from_python
%   'qec'  -> treina CTRL-1, update_Qec_from_python
%   'do'   -> treina CTRL-3, update_DOref_from_python
%   'qw'   -> treina CTRL-4, update_Qw_from_python
% Correr apenas UM agente de cada vez.
%% ===============================
if exist('SIMPLE_AGENT', 'var') && ~isempty(SIMPLE_AGENT)
    AGENT = SIMPLE_AGENT;   % override from base workspace, e.g. SIMPLE_AGENT='do'
else
    AGENT = 'qw';    % <-- default; alternar aqui entre 'qint', 'qec', 'do' e 'qw'
end

if ~ismember(AGENT, {'qint','qec','do','qw'})
    error('[RL_main] AGENT invalido: %s (usar ''qint'', ''qec'', ''do'' ou ''qw'')', AGENT);
end
fprintf('\n[RL_main] Agente activo: %s\n', upper(AGENT));

%% ===============================
% Parametros de tempo
%% ===============================
dt      = 15 / (24*60);   % 15 minutos em dias
tol     = 1e-6;
TIMEOUT = 30;             % segundos a aguardar pelo Python
ABORT_ON_TIMEOUT = true;  % para runs de validacao, falhar cedo e nao continuar com defaults
FAST_SAMPLES_PER_DAY = max(2, round(1.0 / dt));
SRT_WINDOW_DAYS = 3.0;
SRT_WINDOW_SAMPLES = max(FAST_SAMPLES_PER_DAY, round(SRT_WINDOW_DAYS / dt));

if strcmp(AGENT, 'qw')
    agent_dt = 1.0;       % CTRL-4 decide 1x por dia
else
    agent_dt = dt;        % CTRL-1/2/3 mantem 15 min
end

%% ===============================
% Paths dos ficheiros de comunicacao
%% ===============================
FLAG_STATE  = fullfile(COMMS, 'flag_state.run');
FLAG_ACTION = fullfile(COMMS, 'flag_action.run');
STATE_FILE  = fullfile(COMMS, 'state.csv');
ACTION_FILE = fullfile(COMMS, 'action.csv');

%% ===============================
% Limpar flags antigos
%% ===============================
disp('[RL_main] Limpando flags antigos...');
if isfile(FLAG_STATE),  delete(FLAG_STATE);  end
if isfile(FLAG_ACTION), delete(FLAG_ACTION); end

%% ===============================
% Configurar simulacao - desde t=0
%% ===============================
set_param('bsm2_cl', 'SimulationMode', 'accelerator');
set_param('bsm2_cl', 'StartTime',   '0');
if exist('SIMPLE_STOP_TIME', 'var') && ~isempty(SIMPLE_STOP_TIME)
    simple_stop = SIMPLE_STOP_TIME;   % override from base workspace (e.g. short smoke test)
else
    simple_stop = 609;
end
set_param('bsm2_cl', 'StopTime',    num2str(simple_stop));
set_param('bsm2_cl', 'OutputTimes', sprintf('[0:(1/96):%g]', simple_stop));
% PauseTimes is the deterministic pause mechanism on MATLAB versions that
% expose it (e.g. R2019b). On R2025b the model does not, so we silently fall
% back to the manual pause loop below, verified 2026-06-07 to catch both the
% 15-min (do/qec/qint) and the daily (qw) decision boundaries.
if strcmp(AGENT, 'qw')
    try
        set_param('bsm2_cl', 'PauseTimes', '1:1:609');
    catch
    end
else
    try
        set_param('bsm2_cl', 'PauseTimes', '[]');
    catch
    end
end
set_param('bsm2_cl', 'SimulationCommand', 'start');
fprintf('[RL_main] Simulacao iniciada (t=0 -> t=609, agente=%s)...\n', AGENT);

% Aguardar arranque
pause(0.5);
while strcmp(get_param('bsm2_cl', 'SimulationStatus'), 'initializing')
    pause(0.1);
end

% Primeira pausa no proximo instante de decisao
next_pause_time = agent_dt;

%% ===============================
% Loop principal
%% ===============================
while true

    pause(0.05);

    simStatus = get_param('bsm2_cl', 'SimulationStatus');
    if strcmp(simStatus, 'stopped')
        disp('[RL_main] Simulacao finalizada.');
        break
    end

    if strcmp(simStatus, 'running') || strcmp(simStatus, 'paused')

        t_sim = get_param('bsm2_cl', 'SimulationTime');

        if t_sim + tol >= next_pause_time

            if strcmp(get_param('bsm2_cl', 'SimulationStatus'), 'running')
                set_param('bsm2_cl', 'SimulationCommand', 'pause');
            end

            t_p = tic;
            while ~strcmp(get_param('bsm2_cl', 'SimulationStatus'), 'paused')
                pause(0.02);
                if toc(t_p) > 10
                    warning('[RL_main] Timeout a aguardar pausa.');
                    break
                end
            end

            t_sim = get_param('bsm2_cl', 'SimulationTime');
            fprintf('\n[RL_main] t = %.4f dias\n', t_sim);

            %% --- Coleccionar observacoes ---
            if strcmp(AGENT, 'do')
                t_vars = tic;
                while ~(exist('reac4', 'var') == 1 && exist('reac5', 'var') == 1 && ...
                        ~isempty(reac4) && ~isempty(reac5))
                    pause(0.05);
                    if toc(t_vars) > 10
                        error('[RL_main] reac4/reac5 nao ficaram disponiveis no workspace para AGENT=''do''.');
                    end
                end

                SO_4  = reac4(end, 8);    % official BSM2 DO control loop variable (reactor 4)
                SNH_4 = reac4(end, 10);   % ammonia reactor 4
                SNH_5 = reac5(end, 10);   % ammonia reactor 5
                SNO_3 = reac3(end, 9);    % nitrate reactor 3
                % Compatibility note: official BSM2 risk/performance code
                % uses reactor TSS at column 14, but CTRL-3 keeps the
                % legacy particulate-sum proxy below so the already
                % collected baseline statistics and checkpoint remain
                % consistent. If CTRL-3 is retrained from scratch, align
                % this state definition with the official TSS column first.
                TSS_3 = sum(reac3(end, 3:min(7, size(reac3, 2))));

                fprintf('[RL_main] SO4=%.4f SNH4=%.4f SNH5=%.4f SNO3=%.4f TSS3=%.4f\n', ...
                    SO_4, SNH_4, SNH_5, SNO_3, TSS_3);

                %% --- Escrever state.csv (CTRL-3) ---
                T_csv = table(SO_4, SNH_4, SNH_5, SNO_3, TSS_3, t_sim, ...
                    'VariableNames', {'SO_4','SNH_4','SNH_5','SNO_3','TSS_3','time'});
                writetable(T_csv, STATE_FILE);
            elseif strcmp(AGENT, 'qw')
                t_vars = tic;
                while ~(exist('reac1', 'var') == 1 && exist('reac2', 'var') == 1 && ...
                        exist('reac3', 'var') == 1 && exist('reac4', 'var') == 1 && ...
                        exist('reac5', 'var') == 1 && exist('settler', 'var') == 1 && ...
                        exist('VOL1', 'var') == 1 && exist('VOL2', 'var') == 1 && ...
                        exist('VOL3', 'var') == 1 && exist('VOL4', 'var') == 1 && ...
                        exist('VOL5', 'var') == 1 && ...
                        ~isempty(reac1) && ~isempty(reac2) && ~isempty(reac3) && ...
                        ~isempty(reac4) && ~isempty(reac5) && ~isempty(settler))
                    pause(0.05);
                    if toc(t_vars) > 10
                        error(['[RL_main] reac1..5/settler nao ficaram disponiveis ' ...
                               'no workspace para AGENT=''qw''.']);
                    end
                end

                n_day = min([size(reac1,1), size(reac2,1), size(reac3,1), ...
                             size(reac4,1), size(reac5,1), size(settler,1), ...
                             FAST_SAMPLES_PER_DAY]);
                idx_day_r = (size(reac5,1) - n_day + 1):size(reac5,1);
                idx_day_s = (size(settler,1) - n_day + 1):size(settler,1);

                n_srt = min([size(reac1,1), size(reac2,1), size(reac3,1), ...
                             size(reac4,1), size(reac5,1), size(settler,1), ...
                             SRT_WINDOW_SAMPLES]);
                idx_srt_r = (size(reac5,1) - n_srt + 1):size(reac5,1);
                idx_srt_s = (size(settler,1) - n_srt + 1):size(settler,1);

                TSS_5_1d = mean(reac5(idx_day_r, 14));
                % Verified against asm1init_bsm2.m: S_ND is column 11 in
                % the reactor state vectors exported to MATLAB.
                SND_5_1d = mean(reac5(idx_day_r, 11));
                SNH_5_1d = mean(reac5(idx_day_r, 10));
                SNO_3_1d = mean(reac3(idx_day_r, 9));
                EQI_1d = 20648.0 * (30.0 * SNH_5_1d + 10.0 * SNO_3_1d) / 1000.0;
                Qw_mean_1d = mean(settler(idx_day_s, 22));

                TSSvecreactor = ...
                    reac1(idx_srt_r, 14) * VOL1 + ...
                    reac2(idx_srt_r, 14) * VOL2 + ...
                    reac3(idx_srt_r, 14) * VOL3 + ...
                    reac4(idx_srt_r, 14) * VOL4 + ...
                    reac5(idx_srt_r, 14) * VOL5;

                Qwasteflow = settler(idx_srt_s, 22);
                % Verified against perf_risk_bsm2.m:
                %   settler(:,22) -> wastage flow Qw
                %   settler(:,53) -> TSS in wastage sludge
                %   settler(:,36).*settler(:,37) -> effluent TSS mass flow
                TSSwasteconc = settler(idx_srt_s, 53);
                TSSevec2 = settler(idx_srt_s, 36) .* settler(idx_srt_s, 37);
                removed_tss = max(TSSwasteconc .* Qwasteflow + TSSevec2, 1e-6);
                SRT_3d = mean(TSSvecreactor ./ removed_tss);

                fprintf(['[RL_main] TSS5_1d=%.4f SND5_1d=%.4f SRT3d=%.4f ' ...
                         'SNH5_1d=%.4f EQI_1d=%.4f Qw_1d=%.4f\n'], ...
                    TSS_5_1d, SND_5_1d, SRT_3d, SNH_5_1d, EQI_1d, Qw_mean_1d);

                %% --- Escrever state.csv (CTRL-4) ---
                T_csv = table(TSS_5_1d, SND_5_1d, SRT_3d, SNH_5_1d, EQI_1d, ...
                    Qw_mean_1d, SNO_3_1d, t_sim, ...
                    'VariableNames', ...
                    {'TSS_5_1d','SND_5_1d','SRT_3d','SNH_5_1d', ...
                     'EQI_1d','Qw_mean_1d','SNO_3_1d','time'});
                writetable(T_csv, STATE_FILE);
            else
                % As observacoes sao as mesmas para CTRL-1 e CTRL-2:
                % nitrato nos reactores 1/2/3 + racio COD/TN do afluente.
                % reac1/2/3: arrays actualizados pelo Simulink (To Workspace)
                % NAO usar S_NO1/S_NO2 - sao escalares de inicializacao
                SNO_1 = reac1(end, 9);    % nitrato reactor 1 (anoxico)
                SNO_2 = reac2(end, 9);    % nitrato reactor 2 (anoxico)
                SNO_3 = reac3(end, 9);    % nitrato reactor 3 (aerobio)
                SNH_2 = reac2(end, 10);   % amonia reactor 2

                % in: vector 1x21 do influente actual (actualizado)
                SS_in  = in(3);
                SI_in  = in(2);
                SNH_in = in(11);
                CODTN_raw = (SS_in + SI_in) / max(SNH_in, 1.0);   % floor SNH to avoid divide-by-near-zero at init
                CODTN = min(max(CODTN_raw, 0), 100);              % clip to plausible range

                Flow = Qin;
                Temp = T2;

                fprintf('[RL_main] SNO2=%.4f SNO1=%.4f SNO3=%.4f CODTN=%.2f SNH=%.4f\n', ...
                    SNO_2, SNO_1, SNO_3, CODTN, SNH_2);

                %% --- Escrever state.csv (CTRL-1 / CTRL-2) ---
                T_csv = table(SNO_2, SNO_1, SNO_3, CODTN, SNH_2, Flow, Temp, t_sim, ...
                    'VariableNames', ...
                    {'SNO_2','SNO_1','SNO_3','CODTN','SNH_in','Flow','Temp','time'});
                writetable(T_csv, STATE_FILE);
            end

            if isfile(FLAG_STATE), delete(FLAG_STATE); end
            fid = fopen(FLAG_STATE, 'w'); fclose(fid);
            disp('[RL_main] flag_state criado');

            %% --- Aguardar accao do Python ---
            disp('[RL_main] Aguardando accao do Python...');
            t_wait = tic;
            while ~isfile(FLAG_ACTION)
                pause(0.05);
                if toc(t_wait) > TIMEOUT
                    if ABORT_ON_TIMEOUT
                        if isfile(FLAG_STATE), delete(FLAG_STATE); end
                        error(['[RL_main] TIMEOUT a aguardar accao do Python ' ...
                               sprintf('(AGENT=%s, t=%.4f dias). ', AGENT, t_sim) ...
                               'Run invalido para validacao limpa; corrigir o agente ' ...
                               'e recomecar de raiz.']);
                    end
                    warning('[RL_main] TIMEOUT - aplicando default BSM2.');
                    if strcmp(AGENT, 'qint')
                        Tdef = table(61944.0, 'VariableNames', {'Qec'});
                    elseif strcmp(AGENT, 'qec')
                        Tdef = table(2.0,     'VariableNames', {'Qec'});
                    elseif strcmp(AGENT, 'do')
                        Tdef = table(2.0,     'VariableNames', {'Qec'});
                    elseif strcmp(AGENT, 'qw')
                        if t_sim < 182
                            qw_def = 300.0;
                        elseif t_sim < 364
                            qw_def = 450.0;
                        elseif t_sim < 546
                            qw_def = 300.0;
                        else
                            qw_def = 450.0;
                        end
                        Tdef = table(qw_def, 'VariableNames', {'Qec'});
                    end
                    writetable(Tdef, ACTION_FILE);
                    fid = fopen(FLAG_ACTION, 'w'); fclose(fid);
                    break
                end
            end
            disp('[RL_main] Accao recebida');

            %% --- Aplicar accao ---
            % Encaminhamento em funcao do agente activo. Por construcao,
            % so uma destas funcoes e' chamada em cada run - a outra
            % variavel manipulada fica no seu default BSM2.
            if strcmp(AGENT, 'qint')
                update_Qint_from_python(ACTION_FILE);
            elseif strcmp(AGENT, 'qec')
                update_Qec_from_python(ACTION_FILE);
            elseif strcmp(AGENT, 'do')
                update_DOref_from_python(ACTION_FILE);
            elseif strcmp(AGENT, 'qw')
                update_Qw_from_python(ACTION_FILE);
            end

            delete(FLAG_ACTION);
            disp('[RL_main] flag_action consumido');

            %% --- Retomar simulacao ---
            next_pause_time = next_pause_time + agent_dt;
            set_param('bsm2_cl', 'SimulationCommand', 'continue');
            disp('[RL_main] Simulacao retomada');

        end
    end
end

% Clean shutdown: discard the dirty (accelerator-built) model so the headless
% -batch session does not warn about unsaved changes when it exits.
if bdIsLoaded('bsm2_cl')
    try
        close_system('bsm2_cl', 0);
    catch
    end
end
