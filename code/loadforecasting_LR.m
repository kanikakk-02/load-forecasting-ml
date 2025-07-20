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

% Normalize features
X = normalize(X);

data.DayOfWeek = weekday(data.timestamp);
data.HourOfDay = hour(data.timestamp);
X = [X, data.DayOfWeek, data.HourOfDay];

n_lags = 3; 
for i = 1:n_lags
    X = [X, circshift(y, i)];
end
X = X(n_lags+1:end, :);  

y = y(n_lags+1:end);  

% Split dataset into training and testing sets
train_ratio = 0.8;
n_train = round(length(y) * train_ratio);
X_train = X(1:n_train, :);
y_train = y(1:n_train);
X_test = X(n_train+1:end, :);
y_test = y(n_train+1:end);

% Convert X_train and X_test into tables with proper column names
X_train_table = array2table(X_train, 'VariableNames', [features, "DayOfWeek", "HourOfDay", "Lag1", "Lag2", "Lag3"]);
X_test_table = array2table(X_test, 'VariableNames', [features, "DayOfWeek", "HourOfDay", "Lag1", "Lag2", "Lag3"]);

% Train a Simple Linear Regression Model
disp("Training Linear Regression Model...");
Mdl = fitlm(X_train_table, y_train);

% Make predictions
y_pred = predict(Mdl, X_test_table);

% Evaluate the model
rmse = sqrt(mean((y_test - y_pred).^2));
r2_score = 1 - sum((y_test - y_pred).^2) / sum((y_test - mean(y_test)).^2);
disp("Model Evaluation:");
disp("RMSE: " + rmse);
disp("R^2 Score: " + r2_score);

% Plot actual vs predicted load
figure;
plot(data.timestamp(n_train+n_lags+1:end), y_test, 'b', 'DisplayName', 'Actual Load'); % Adjust time range to match lag
hold on;
plot(data.timestamp(n_train+n_lags+1:end), y_pred, 'r', 'DisplayName', 'Predicted Load'); % Adjust time range to match lag
legend;
title('Actual vs Predicted Electric Load (Linear Regression)');
xlabel('Time');
ylabel('Electric Load (MW)');
hold off;

% Save the trained model
save('LoadForecastModel_LinearRegression.mat', 'Mdl');
