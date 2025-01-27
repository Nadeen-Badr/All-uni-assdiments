CREATE TABLE Users (
    UserID INT PRIMARY KEY AUTO_INCREMENT,
    Username VARCHAR(50) NOT NULL,
    Email VARCHAR(100) NOT NULL,
    PasswordHash VARCHAR(255) NOT NULL,
   
);

-- Create Buildings table
CREATE TABLE Buildings (
    BuildingID INT PRIMARY KEY AUTO_INCREMENT,
    Title VARCHAR(100) NOT NULL,
    Location VARCHAR(255) NOT NULL,
    Size INT,
    Features TEXT,
    StartingBid DECIMAL(10, 2) NOT NULL,
    GoalAmount DECIMAL(10, 2) NOT NULL,
    CurrnentAmount DECIMAL(10, 2) NOT NULL,
    AuctionEndTime TIMESTAMP,
   
);

-- Create Subscriptions table
CREATE TABLE Subscriptions (
    SubscriptionID INT PRIMARY KEY AUTO_INCREMENT,
    UserID INT,
    BuildingID INT,
    AmountContributed DECIMAL(10, 2) NOT NULL,
    SubscriptionTime TIMESTAMP,
    FOREIGN KEY (UserID) REFERENCES Users(UserID),
    FOREIGN KEY (BuildingID) REFERENCES Buildings(BuildingID)
);

-- Create Transactions table
CREATE TABLE Transactions (
    TransactionID INT PRIMARY KEY AUTO_INCREMENT,
    UserID INT,
    BuildingID INT,
    Amount DECIMAL(10, 2) NOT NULL,
    TransactionTime TIMESTAMP,
    FOREIGN KEY (UserID) REFERENCES Users(UserID),
    FOREIGN KEY (BuildingID) REFERENCES Buildings(BuildingID)
);