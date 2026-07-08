function update_DOref_from_python(action_file)
% UPDATE_DOREF_FROM_PYTHON
%   Le o setpoint de oxigenio dissolvido (DOref) do action.csv escrito
%   pelo Python e aplica-o a variavel SO4ref do workspace MATLAB.
%
%   Nota de nomenclatura: a coluna no CSV chama-se 'Qec' por razoes
%   historicas de compatibilidade do ficheiro de comunicacao
%   partilhado. O valor e' interpretado aqui como SO4ref / DOref do
%   loop oficial de oxigenio do reactor 4 aerobio, NAO como carbono
%   externo.
%
%   SO4ref e' o nome canonico do BSM2 para o setpoint de oxigenio
%   dissolvido consumido pelo bloco SO4_control no subsistema
%   ActivatedSludge. No benchmark oficial, esse loop regula o reactor 4;
%   o mesmo sinal KLa afecta o reactor 5 via KLa4/2.
%
%   Limites usados para CTRL-3:
%       DOref in [0, 10] mg/L
%
%   Default BSM2 (Section 9, "Initialization"): DOref = 2.0 mg/L.

    DOREF_MIN = 0.0;
    DOREF_MAX = 10.0;

    try
        data = readmatrix(action_file);
        data = data(~isnan(data));
        DOref = data(end);   % coluna 'Qec' por compatibilidade historica
        DOref = max(DOREF_MIN, min(DOREF_MAX, DOref));
    catch e
        warning('update_DOref_from_python: %s - usando default BSM2 (2.0 mg/L)', e.message);
        DOref = 2.0;
    end

    % Aplica ao loop oficial SO4_control (reactor 4; reactor 5 acoplado via KLa4/2)
    assignin('base', 'SO4ref', DOref);

    fprintf('[RL] DOref = %.3f mg/L (SO4ref)\n', DOref);

end
