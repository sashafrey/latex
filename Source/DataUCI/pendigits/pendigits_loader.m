function pendigits_loader
    XTrain = dlmread('pendigits_dyn_train.csv', ',');
    XTest = dlmread('pendigits_dyn_test.csv', ',');
    YTrain = dlmread('pendigits_label_train.csv', ',');
    YTest = dlmread('pendigits_label_test.csv', ',');
    X_full = [XTrain; XTest];
    Y_full = 2*double([YTrain; YTest] <= 5) - 1;
    save('pendigits_data', 'X_full', 'Y_full');
end