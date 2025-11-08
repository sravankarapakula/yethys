-- Add missing columns safely
SET @dbname = DATABASE();
SET @tablename = "organizations";

-- Add views column
SET @col_exists = (SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS 
    WHERE TABLE_SCHEMA = @dbname AND TABLE_NAME = @tablename AND COLUMN_NAME = 'views');
SET @query = IF(@col_exists = 0, 
    'ALTER TABLE organizations ADD COLUMN views INT DEFAULT 0',
    'SELECT "views column already exists"');
PREPARE stmt FROM @query;
EXECUTE stmt;

-- Add verified column
SET @col_exists = (SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS 
    WHERE TABLE_SCHEMA = @dbname AND TABLE_NAME = @tablename AND COLUMN_NAME = 'verified');
SET @query = IF(@col_exists = 0,
    'ALTER TABLE organizations ADD COLUMN verified BOOLEAN DEFAULT FALSE',
    'SELECT "verified column already exists"');
PREPARE stmt FROM @query;
EXECUTE stmt;

-- Create interactions table
CREATE TABLE IF NOT EXISTS interactions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    organization_id INT NOT NULL,
    action_type ENUM('view', 'shortlist', 'contact', 'reject') NOT NULL,
    session_id VARCHAR(100),
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_user (user_id),
    INDEX idx_org (organization_id),
    INDEX idx_action (action_type)
);

SELECT "✅ Database setup complete!" as status;
