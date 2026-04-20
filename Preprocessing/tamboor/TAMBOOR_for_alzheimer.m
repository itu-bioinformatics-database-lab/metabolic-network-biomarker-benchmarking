clear all; clc;

% Load required data prepared before
load('indexs_mtb.mat');   % indexes of all exchange reactions
load('pro_ind.mat');      % indexes of exchange reactions with positive maximum secretion rate (producible metabolites)
load('pro_mtb.mat');      % metabolite IDs of produced metabolites
load('met_names.mat');    % names of produced metabolites
load('max_pro.mat');      % maximum secretion rate of given reactions
load('Level_Data.mat');   % includes reaction scores for control and disease cases, and fold changes (FCs) for comparisons

% Default fold changes
default_fc = ones(13069,1);
default_fc(indexs_mtb) = 2;

% Single example FC from Level_Data
fc = Level_Data.FC;  % Assuming Level_Data contains a single entry with FC data

timbr_scores = zeros(length(pro_ind), 2); % Initialize TIMBR scores with zeros for both conditions
modified_timbr_scores = zeros(length(pro_ind), 2); % Initialize modified TIMBR scores with zeros for both conditions
% Perform calculations for both conditions (Control and Disease)
for j = 1:2
    % Load model with required constraints
    load('model_constrain.mat')
    
    % Set fold change values based on condition
    if j == 1
        model.fc = default_fc .* fc;  % Control condition
    else
        model.fc = default_fc ./ fc;  % Disease condition
    end

    % Convert model to irreversible format
    [modelIrrev, matchRev, rev2irrev, irrev2rev] = convertToIrreversible(model);
    model = modelIrrev;

    % Prepare model for Gurobi format
    model.A = sparse(model.S);  % Stoichiometric matrix as A
    model.rhs = zeros(length(model.A(:,1)),1);  % Right-hand side constraints
    model.obj = zeros(length(model.A(1,:)),1);  % Initialize objective function
    model.sense = ['='];
    model.vtype = 'C';

    % Calculate TAMBOOR score for each producible metabolite
    rxn_act5 = [];
    for k = 1:length(pro_ind)
        model2 = model;
        model2.lb(pro_ind(k)) = max_pro(k) * 0.9;  % Adjust lower bound for secretion
        model2.obj = model2.fc;  % Set objective function
        rslt = gurobi(model2);  % Run Gurobi optimization
        
        % TAMBOOR score: count reactions with flux > 0.00001
        rxn_act5(k) = length(find(rslt.x > 0.00001));
    
        % TIMBR score: use the objective value from Gurobi results
        timbr_scores(k, j) = rslt.objval;
        
        % Modified TIMBR score: sum of fluxes for internal reactions
        % modified_timbr_scores(k, j) = sum(rslt.x(find(model2.exch == 1)));
        
        % Cobra function to calculate TIMBR score
        exchRxns = findExcRxns(model2);  % Identify exchange reactions
        modified_timbr_scores(k, j) = sum(rslt.x(exchRxns));  % Sum fluxes of exchange reactions
    end
    act5(:,j) = rxn_act5';
end

% Compute the TAMBOOR score and Z-score
scores = act5(1:end,:);
raw_score = (scores(:,1) - scores(:,2)) ./ (scores(:,1) + scores(:,2));
raw_timbr_score = (timbr_scores(:,1) - timbr_scores(:,2)) ./ (timbr_scores(:,1) + timbr_scores(:,2));
raw_modified_timbr_score = (modified_timbr_scores(:,1) - modified_timbr_scores(:,2)) ./ (modified_timbr_scores(:,1) + modified_timbr_scores(:,2));
ort = mean(raw_score);
st_d = std(raw_score);
z_score = (raw_score - ort) ./ st_d;
act5(:,3) = z_score;
act5(:,4) = raw_score;
timbr_scores(:,3) = raw_timbr_score;
modified_timbr_scores(:,3) = raw_modified_timbr_score;

% Column 1: Control, Column 2: Disease, Column 3: Z-score, Column 4: Raw score

% Store results in ScoreData
ScoreData.TAMBOORScore = act5;
ScoreData.TIMBRScore = timbr_scores;
ScoreData.ModifiedTIMBRScore = modified_timbr_scores;
ScoreData.DataID = Level_Data.DataID;  % Assuming Level_Data has a DataID field
ScoreData.metNames = mtb;

save('ScoreData.mat', 'ScoreData');