%% BSM2 Closed-Loop Correlation Analysis - Versão Expandida
% Compatível com todas as versões MATLAB (usa imagesc em vez de heatmap)
% Evaluation period: days 245-609 (BSM2 standard)

%% --- Configuration ---
t1 = 245;
t2 = 609;
idx = find(t >= t1 & t <= t2);
N   = length(idx);

fprintf('Evaluation period: day %d to %d\n', t1, t2);
fprintf('Number of samples: %d\n\n', N);

%% --- Helper: extrai coluna e força vector coluna de comprimento N ---
get_col = @(mat, c) reshape(mat(idx, min(c, size(mat,2))), N, 1);

%% ================================================================
%  EXTRACÇÃO DE VARIÁVEIS
%  ASM1: 1=SI 2=SS 3=XI 4=XS 5=XBH 6=XBA 7=XP
%        8=SO 9=SNO 10=SNH 11=SND 12=XND 13=SALK 15=T
%  Influente: 3=SS 10=SNO 11=SNH 16=Q 17=T
%% ================================================================

%% Influente
Q_in   = get_col(DYNINFLUENT_BSM2, 16);
SNH_in = get_col(DYNINFLUENT_BSM2, 11);
SS_in  = get_col(DYNINFLUENT_BSM2,  3);
SNO_in = get_col(DYNINFLUENT_BSM2, 10);

%% Zona 1 (anóxica)
SS_1   = get_col(reac1,  2);
XBH_1  = get_col(reac1,  5);
XBA_1  = get_col(reac1,  6);
SO_1   = get_col(reac1,  8);
SNO_1  = get_col(reac1,  9);
SNH_1  = get_col(reac1, 10);
SND_1  = get_col(reac1, 11);
SALK_1 = get_col(reac1, 13);
T      = get_col(reac1, 15);
TSS_1  = sum(reac1(idx, 3:min(7,size(reac1,2))), 2);

%% Zona 2 (anóxica)
SS_2   = get_col(reac2,  2);
XBH_2  = get_col(reac2,  5);
SO_2   = get_col(reac2,  8);
SNO_2  = get_col(reac2,  9);   % TARGET CTRL-1/2
SNH_2  = get_col(reac2, 10);
SND_2  = get_col(reac2, 11);
SALK_2 = get_col(reac2, 13);
TSS_2  = sum(reac2(idx, 3:min(7,size(reac2,2))), 2);

%% Zona 3 (aeróbica)
SS_3   = get_col(reac3,  2);
XBH_3  = get_col(reac3,  5);
XBA_3  = get_col(reac3,  6);
SO_3   = get_col(reac3,  8);
SNO_3  = get_col(reac3,  9);
SNH_3  = get_col(reac3, 10);
SND_3  = get_col(reac3, 11);
SALK_3 = get_col(reac3, 13);
TSS_3  = sum(reac3(idx, 3:min(7,size(reac3,2))), 2);

%% Zona 4 (aeróbica)
SS_4   = get_col(reac4,  2);
SO_4   = get_col(reac4,  8);
SNO_4  = get_col(reac4,  9);
SNH_4  = get_col(reac4, 10);
SALK_4 = get_col(reac4, 13);
TSS_4  = sum(reac4(idx, 3:min(7,size(reac4,2))), 2);

%% Zona 5 (aeróbica)
SS_5   = get_col(reac5,  2);
XBH_5  = get_col(reac5,  5);
XBA_5  = get_col(reac5,  6);
SO_5   = get_col(reac5,  8);   % TARGET CTRL-3
SNO_5  = get_col(reac5,  9);
SNH_5  = get_col(reac5, 10);
SND_5  = get_col(reac5, 11);
SALK_5 = get_col(reac5, 13);
TSS_5  = sum(reac5(idx, 3:min(7,size(reac5,2))), 2);   % TARGET CTRL-4

%% Variáveis derivadas
COD_TN    = SS_1  ./ max(SNH_1 + SNO_1, 1e-6);
N_load    = Q_in  .* (SNH_in + SNO_in);
dSNO_13   = SNO_3 - SNO_1;
dSNO_35   = SNO_5 - SNO_3;
NitFrac_3 = XBA_3 ./ max(XBA_3 + XBH_3, 1e-6);

%% ================================================================
%  CONSTRUÇÃO SEGURA DA MATRIZ X
%% ================================================================
all_vars = { ...
    Q_in,      'Q_{in}';        SNH_in,    'SNH_{in}'; ...
    SS_in,     'SS_{in}';       SNO_in,    'SNO_{in}'; ...
    N_load,    'N_{load}'; ...
    SS_1,      'SS_1';          SNO_1,     'SNO_1'; ...
    SNH_1,     'SNH_1';         SO_1,      'SO_1'; ...
    SALK_1,    'SALK_1';        TSS_1,     'TSS_1'; ...
    SND_1,     'SND_1';         XBH_1,     'XBH_1'; ...
    XBA_1,     'XBA_1'; ...
    SS_2,      'SS_2';          SNH_2,     'SNH_2'; ...
    SO_2,      'SO_2';          SALK_2,    'SALK_2'; ...
    TSS_2,     'TSS_2';         SND_2,     'SND_2'; ...
    XBH_2,     'XBH_2'; ...
    SS_3,      'SS_3';          SNO_3,     'SNO_3'; ...
    SNH_3,     'SNH_3';         SO_3,      'SO_3'; ...
    SALK_3,    'SALK_3';        TSS_3,     'TSS_3'; ...
    SND_3,     'SND_3';         XBH_3,     'XBH_3'; ...
    XBA_3,     'XBA_3'; ...
    SS_4,      'SS_4';          SNO_4,     'SNO_4'; ...
    SNH_4,     'SNH_4';         SO_4,      'SO_4'; ...
    SALK_4,    'SALK_4';        TSS_4,     'TSS_4'; ...
    SS_5,      'SS_5';          SNO_5,     'SNO_5'; ...
    SNH_5,     'SNH_5';         SALK_5,    'SALK_5'; ...
    SND_5,     'SND_5';         XBH_5,     'XBH_5'; ...
    XBA_5,     'XBA_5'; ...
    COD_TN,    'COD_{TN}';      dSNO_13,   'dSNO_{13}'; ...
    dSNO_35,   'dSNO_{35}';     NitFrac_3, 'NitFrac_3'; ...
    T,         'T' ...
};

nCand     = size(all_vars, 1);
X         = nan(N, nCand);
obs_names = cell(1, nCand);
n_ok      = 0;

for k = 1:nCand
    v    = all_vars{k,1}(:);   % força coluna
    name = all_vars{k,2};
    if length(v) == N
        n_ok            = n_ok + 1;
        X(:, n_ok)      = v;
        obs_names{n_ok} = name;
    else
        fprintf('AVISO: "%s" ignorada (length=%d != %d)\n', name, length(v), N);
    end
end

X         = X(:, 1:n_ok);
obs_names = obs_names(1:n_ok);
fprintf('Variáveis carregadas: %d / %d\n\n', n_ok, nCand);

%% --- Targets ---
targets     = [SNO_2, SO_5, TSS_5];
ctrl_labels = {'CTRL-1/2 (SNO_2)', 'CTRL-3 (SO_5)', 'CTRL-4 (TSS_5)'};

%% --- Remover constantes / maioritariamente NaN ---
valid = true(1, n_ok);
for i = 1:n_ok
    col = X(:,i);
    if mean(isnan(col)) > 0.5 || std(col(isfinite(col))) < 1e-12
        valid(i) = false;
        fprintf('Removida (constante/NaN): %s\n', obs_names{i});
    end
end
X         = X(:, valid);
obs_names = obs_names(valid);
nObs      = size(X, 2);
nCtrl     = size(targets, 2);
fprintf('\nVariáveis retidas: %d\n\n', nObs);

%% --- z-score ---
for i = 1:nObs
    col = X(:,i);
    m   = mean(col, 'omitnan');
    s   = std(col,  'omitnan');
    if s > 0, X(:,i) = (col - m) / s; end
end

%% --- Pearson correlation ---
R = nan(nObs, nCtrl);
for j = 1:nCtrl
    y = targets(:,j);
    for i = 1:nObs
        x  = X(:,i);
        ok = isfinite(x) & isfinite(y);
        if nnz(ok) > 1 && std(x(ok)) > 0 && std(y(ok)) > 0
            C      = corrcoef(x(ok), y(ok));
            R(i,j) = C(1,2);
        end
    end
end

%% --- Ordenar por max |r| ---
[~, ord]         = sort(max(abs(R), [], 2), 'descend');
R_sorted         = R(ord, :);
obs_names_sorted = obs_names(ord);

%% --- Print top-5 por agente ---
for j = 1:nCtrl
    vals = abs(R(:,j));
    vals(isnan(vals)) = -inf;
    [~, top_idx] = sort(vals, 'descend');
    fprintf('%s\n', ctrl_labels{j});
    for k = 1:min(5, numel(top_idx))
        i = top_idx(k);
        fprintf('  %-28s  r = %+.4f\n', obs_names{i}, R(i,j));
    end
    fprintf('\n');
end

%% ================================================================
%  HEATMAP CUSTOMIZADO COM imagesc
%  Compatível com todas as versões MATLAB
%% ================================================================

cell_h   = 26;                          % altura de cada célula (px)
margin_l = 120;                         % margem esquerda para labels Y
margin_b = 60;                          % margem inferior para labels X
fig_w    = 700;
fig_h    = nObs * cell_h + margin_b + 80;

figure('Color', [0 0 0], ...
       'Position', [50 30 fig_w max(fig_h, 400)]);

ax = axes('Position', [margin_l/fig_w, margin_b/fig_h, ...
                        0.72, (nObs*cell_h)/fig_h], ...
          'Color', 'k');

imagesc(R_sorted, [-1 1]);
colormap(ax, redblue_colormap());
cb = colorbar(ax);
cb.Color = 'w';
cb.FontSize = 9;

%% Labels eixo X (targets)
ax.XTick      = 1:nCtrl;
ax.XTickLabel = ctrl_labels;
ax.XTickLabelRotation = 0;
ax.XAxisLocation = 'bottom';
ax.FontSize   = 9;
ax.XColor     = 'w';
ax.YColor     = 'w';
ax.TickLength = [0 0];

%% Labels eixo Y (observações) — ordem de cima para baixo
ax.YTick      = 1:nObs;
ax.YTickLabel = obs_names_sorted;

%% Grid entre células
hold(ax, 'on');
for i = 0.5:1:nObs+0.5
    plot(ax, [0.5, nCtrl+0.5], [i i], 'k-', 'LineWidth', 0.4);
end
for j = 0.5:1:nCtrl+0.5
    plot(ax, [j j], [0.5, nObs+0.5], 'k-', 'LineWidth', 0.4);
end

%% Texto nas células
for i = 1:nObs
    for j = 1:nCtrl
        val = R_sorted(i,j);
        if ~isnan(val)
            if abs(val) >= 0.7
                lbl = sprintf('%.3f *', val);
                fw  = 'bold';
            else
                lbl = sprintf('%.3f', val);
                fw  = 'normal';
            end
            % cor do texto: preto em células muito saturadas, branco no resto
            if abs(val) > 0.85
                tc = [1 1 1];
            else
                tc = [0.15 0.15 0.15];
            end
            text(ax, j, i, lbl, ...
                'HorizontalAlignment', 'center', ...
                'VerticalAlignment',   'middle', ...
                'FontSize', 8, ...
                'FontWeight', fw, ...
                'Color', tc);
        end
    end
end

title(ax, 'Pearson Correlation: Observations vs Control Targets', ...
    'Color', 'w', 'FontSize', 12, 'FontWeight', 'bold');

%% --- Export ---
out_dir = fullfile(pwd, 'output');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

exportgraphics(gcf, fullfile(out_dir, 'bsm2_corr_heatmap_full.png'), 'Resolution', 300);

valid_ctrl = matlab.lang.makeValidName(ctrl_labels);
valid_obs  = matlab.lang.makeValidName(obs_names_sorted);
corrT = array2table(R_sorted, 'VariableNames', valid_ctrl, 'RowNames', valid_obs);
writetable(corrT, fullfile(out_dir, 'bsm2_corr_results_full.csv'),  'WriteRowNames', true);
writetable(corrT, fullfile(out_dir, 'bsm2_corr_results_full.xlsx'), ...
    'Sheet', 'Correlation', 'WriteRowNames', true);

fprintf('Export completo -> %s\n', out_dir);

%% --- Local function ---
function cmap = redblue_colormap()
    n    = 256;
    half = n / 2;
    cmap = [linspace(0,1,half)', linspace(0,1,half)', ones(half,1); ...
            ones(half,1), linspace(1,0,half)', linspace(1,0,half)'];
end