data = readtable('load_forecasting_dataset.csv');

disp("Column names in the dataset:");
disp(data.Properties.VariableNames);

data.YEAR = str2double(string(data.YEAR));
data.Month = str2double(string(data.Month));
data.Day = str2double(string(data.Day));
data.Hour = str2double(string(data.Hour));

data.timestamp = datetime(data.YEAR, data.Month, data.Day, data.Hour, 0, 0);

data = movevars(data, 'timestamp', 'Before', 1);

data(:, {'YEAR', 'Month', 'Day', 'Hour'}) = [];

data = sortrows(data, 'timestamp');

disp("Preview of the data:");
disp(head(data));

target = "ElectricLoad_MW_";
features = {'irradiance', 'temperature', 'dewpoint', 'specificHumidity', 'windSpeed'};

if ~ismember(target, data.Properties.VariableNames)
    error("Target column '%s' not found in dataset. Please verify column names.", target);
end

y = data.(target); % Dependent variable
X = data{:, features}; % Features matrix

X = fillmissing(X, 'previous'); 

% Normalizing features
X = normalize(X);

data.DayOfWeek = weekday(data.timestamp);
data.HourOfDay = hour(data.timestamp);
X = [X, data.DayOfWeek, data.HourOfDay];

n_lags = 3; % Adding lag of 3 hours as a feature
for i = 1:n_lags
    X = [X, circshift(y, i)];
end
X = X(n_lags+1:end, :);  

y = y(n_lags+1:end); 

% Splitting dataset into training and testing sets
train_ratio = 0.8;
n_train = round(length(y) * train_ratio);
X_train = X(1:n_train, :);
y_train = y(1:n_train);
X_test = X(n_train+1:end, :);
y_test = y(n_train+1:end);

cv = cvpartition(length(y_train), 'KFold', 5);

% Training a Random Forest model with cross-validation
Mdl = fitrensemble(X_train, y_train, 'Method', 'Bag', 'NumLearningCycles', 200, 'CrossVal', 'on', 'CVPartition', cv);

y_pred = predict(Mdl.Trained{1}, X_test);  % Use the first model from cross-validation

rmse = sqrt(mean((y_test - y_pred).^2));
r2_score = 1 - sum((y_test - y_pred).^2) / sum((y_test - mean(y_test)).^2);
disp("Model Evaluation:");
disp("RMSE: " + rmse);
disp("R^2 Score: " + r2_score);

figure;
plot(data.timestamp(n_train+n_lags+1:end), y_test, 'b', 'DisplayName', 'Actual Load'); % Adjust time range to match lag
hold on;
plot(data.timestamp(n_train+n_lags+1:end), y_pred, 'r', 'DisplayName', 'Predicted Load'); % Adjust time range to match lag
legend;
title('Actual vs Predicted Electric Load');
xlabel('Time');
ylabel('Electric Load (MW)');
hold off;

n_forecast = 48;
X_forecast = X(end-n_forecast+1:end, :);
y_forecast = predict(Mdl.Trained{1}, X_forecast);

time_forecast = data.timestamp(n_train+1:end);
time_forecast = time_forecast(end-n_forecast+1:end);

figure;
plot(time_forecast, y_forecast, 'g', 'DisplayName', 'Forecast');
legend;
title('Random Forest Forecast for Electric Load (Next 48 Hours)');
xlabel('Time');
ylabel('Electric Load (MW)');
hold off;

save('LoadForecastModel.mat', 'Mdl');
