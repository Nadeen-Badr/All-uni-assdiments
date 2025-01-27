BEGIN;

CREATE SEQUENCE activities_activity_id_seq START 1;

CREATE TABLE IF NOT EXISTS activities
(
    activity_id SERIAL PRIMARY KEY,
    activity_text VARCHAR(255) NOT NULL,
    icon_url TEXT,
    CONSTRAINT activities_activity_id_seq_unique UNIQUE (activity_id)
);

CREATE SEQUENCE activity_recommendations_recommendation_id_seq START 1;

CREATE TABLE IF NOT EXISTS activity_recommendations
(
    recommendation_id SERIAL PRIMARY KEY,
    type VARCHAR(50),
    activity_name VARCHAR(255) NOT NULL,
    CONSTRAINT activity_recommendations_activity_name_key UNIQUE (activity_name)
);

CREATE SEQUENCE cbt_questions_question_id_seq START 1;

CREATE TABLE IF NOT EXISTS cbt_questions
(
    question_id SERIAL PRIMARY KEY,
    type_id INTEGER,
    question_text TEXT NOT NULL,
    CONSTRAINT fk_cbt_types FOREIGN KEY (type_id) REFERENCES cbt_types(type_id)
);

CREATE SEQUENCE cbt_types_type_id_seq START 1;

CREATE TABLE IF NOT EXISTS cbt_types
(
    type_id SERIAL PRIMARY KEY,
    type_name VARCHAR(255) NOT NULL,
    icon_url TEXT,
    type_info TEXT
);

CREATE SEQUENCE daily_emotion_tracks_track_id_seq START 1;

CREATE TABLE IF NOT EXISTS daily_emotionStress_tracks
(
    track_id SERIAL PRIMARY KEY,
    user_id INTEGER,
    emotion_id INTEGER,
    sub_emotion_id INTEGER,
    has_stress BOOLEAN,
    track_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    additional_notes TEXT,
    CONSTRAINT fk_emotion FOREIGN KEY (emotion_id) REFERENCES emotions(emotion_id),
    CONSTRAINT fk_sub_emotion FOREIGN KEY (sub_emotion_id) REFERENCES sub_emotions(sub_emotion_id),
    CONSTRAINT daily_emotionStress_tracks_user_id_fkey FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE SEQUENCE daily_emotion_tracks_track_id_seq START 1;

CREATE TABLE IF NOT EXISTS daily_activity_tracks
(
    track_id SERIAL PRIMARY KEY,
    user_id INTEGER,
    reason_id INTEGER,
    activity_id INTEGER,
    CONSTRAINT fk_reason FOREIGN KEY (reason_id) REFERENCES reasons(reason_id),
    CONSTRAINT fk_activity FOREIGN KEY (activity_id) REFERENCES activities(activity_id),
    CONSTRAINT daily_activity_tracks_user_id_fkey FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE SEQUENCE depression_test_scores_score_id_seq START 1;

CREATE TABLE IF NOT EXISTS depression_test_scores
(
    score_id SERIAL PRIMARY KEY,
    user_id INTEGER,
    score INTEGER NOT NULL,
    test_date TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT user_Metadata_user_id_fkey FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE SEQUENCE depression_test_testquestion_id_seq START 1;

CREATE TABLE IF NOT EXISTS depression_test_testquestion
(
    id SERIAL PRIMARY KEY,
    question VARCHAR(255) NOT NULL
);

CREATE SEQUENCE depression_test_testquestion_answer_options_id_seq START 1;

CREATE TABLE IF NOT EXISTS depression_test_testquestion_answer_options
(
    id SERIAL PRIMARY KEY,
    testquestion_id INTEGER NOT NULL,
    answeroption_id INTEGER NOT NULL,
    CONSTRAINT depression_test_testques_testquestion_id_answerop_020f5d71_uniq UNIQUE (testquestion_id, answeroption_id),
    CONSTRAINT fk_testquestion FOREIGN KEY (testquestion_id) REFERENCES depression_test_testquestion(id),
    CONSTRAINT fk_answeroption FOREIGN KEY (answeroption_id) REFERENCES depression_test_answeroption(id)
);

CREATE SEQUENCE depression_test_answeroption_id_seq START 1;

CREATE TABLE IF NOT EXISTS depression_test_answeroption
(
    id SERIAL PRIMARY KEY,
    value INTEGER NOT NULL,
    label VARCHAR(255) NOT NULL
);

CREATE SEQUENCE emotions_emotion_id_seq START 1;

CREATE TABLE IF NOT EXISTS emotions
(
    emotion_id SERIAL PRIMARY KEY,
    emotion_name VARCHAR(255) NOT NULL,
    icon_url TEXT
);

CREATE SEQUENCE userplan_activities_user_activity_id_seq START 1;

CREATE TABLE IF NOT EXISTS userplan_activities
(
    user_activity_id SERIAL PRIMARY KEY,
    user_id INTEGER,
    activity_id INTEGER,
    CONSTRAINT fk_userplan_activities_user_id FOREIGN KEY (user_id) REFERENCES users(user_id),
    CONSTRAINT fk_userplan_activities_activity_id FOREIGN KEY (activity_id) REFERENCES plan_activities(activity_id)
);

CREATE SEQUENCE plan_activities_activity_id_seq START 1;

CREATE TABLE IF NOT EXISTS plan_activities
(
    activity_id SERIAL PRIMARY KEY,
    topic_id INTEGER,
    activity_description TEXT NOT NULL,
    CONSTRAINT fk_plan_activities_topic_id FOREIGN KEY (topic_id) REFERENCES topics(topic_id)
);

CREATE SEQUENCE preferences_questions_question_id_seq START 1;

CREATE TABLE IF NOT EXISTS preferences_questions
(
    question_id SERIAL PRIMARY KEY,
    question_text TEXT NOT NULL,
    tag VARCHAR(255),
    CONSTRAINT fk_tag FOREIGN KEY (tag) REFERENCES tags(tag)
);

CREATE SEQUENCE user_preferences_answers_answer_id_seq START 1;

CREATE TABLE IF NOT EXISTS user_preferences_answers
(
    answer_id SERIAL PRIMARY KEY,
    user_id INTEGER,
    tag VARCHAR(255),
    answer BOOLEAN NOT NULL,
    CONSTRAINT fk_user_preferences_answers_user_id FOREIGN KEY (user_id) REFERENCES users(user_id),
    CONSTRAINT fk_user_preferences_answers_tag FOREIGN KEY (tag) REFERENCES preferences_questions(tag)
);

CREATE SEQUENCE reasons_reason_id_seq START 1;

CREATE TABLE IF NOT EXISTS reasons
(
    reason_id SERIAL PRIMARY KEY,
    reason_text VARCHAR(255) NOT NULL,
    icon_url TEXT
);

CREATE SEQUENCE sub_emotions_sub_emotion_id_seq START 1;

CREATE TABLE IF NOT EXISTS sub_emotions
(
    sub_emotion_id SERIAL PRIMARY KEY,
    emotion_id INTEGER,
    sub_emotion_name VARCHAR(255) NOT NULL,
    icon_url TEXT,
    description TEXT,
    tips TEXT,
    CONSTRAINT fk_sub_emotion_emotion_id FOREIGN KEY (emotion_id) REFERENCES emotions(emotion_id)
);

CREATE SEQUENCE topics_topic_id_seq START 1;

CREATE TABLE IF NOT EXISTS topics
(
    topic_id SERIAL PRIMARY KEY,
    topic_name VARCHAR(255) NOT NULL
);

CREATE SEQUENCE learning_topics_learning_topic_id_seq START 1;

CREATE TABLE IF NOT EXISTS learning_topics
(
    learning_topic_id SERIAL PRIMARY KEY,
    topic_name VARCHAR(255) NOT NULL
);

CREATE SEQUENCE lessons_lesson_id_seq START 1;

CREATE TABLE IF NOT EXISTS lessons
(
    lesson_id SERIAL PRIMARY KEY,
    learning_topic_id INTEGER,
    lesson_title VARCHAR(255) NOT NULL,
    lesson_content TEXT NOT NULL,
    CONSTRAINT fk_lessons_learning_topic_id FOREIGN KEY (learning_topic_id) REFERENCES learning_topics(learning_topic_id)
);

CREATE SEQUENCE user_settings_setting_id_seq START 1;

CREATE TABLE IF NOT EXISTS user_Metadata
(
    setting_id SERIAL PRIMARY KEY,
    user_id INTEGER UNIQUE,
    notifications_enabled BOOLEAN DEFAULT true,
    start_day_time TIME WITHOUT TIME ZONE,
    image_url TEXT,
    date_of_birth TIMESTAMP WITH TIME ZONE,
    gender VARCHAR(50),
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    CONSTRAINT user_Metadata_user_id_fkey FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE SEQUENCE user_topics_user_topic_id_seq START 1;

CREATE TABLE IF NOT EXISTS user_topics
(
    user_topic_id SERIAL PRIMARY KEY,
    user_id INTEGER,
    topic_id INTEGER,
    CONSTRAINT fk_user_topics_user_id FOREIGN KEY (user_id) REFERENCES users(user_id),
    CONSTRAINT fk_user_topics_topic_id FOREIGN KEY (topic_id) REFERENCES topics(topic_id)
);

CREATE SEQUENCE user_weekly_answers_answer_id_seq START 1;

CREATE TABLE IF NOT EXISTS user_weekly_answers
(
    answer_id SERIAL PRIMARY KEY,
    user_id INTEGER,
    question_id INTEGER,
    answer INTEGER,
    answer_date TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_user_weekly_answers_user_id FOREIGN KEY (user_id) REFERENCES users(user_id),
    CONSTRAINT fk_user_weekly_answers_question_id FOREIGN KEY (question_id) REFERENCES weekly_checking_questions(question_id)
);

CREATE SEQUENCE weekly_checking_questions_question_id_seq START 1;

CREATE TABLE IF NOT EXISTS weekly_checking_questions
(
    question_id SERIAL PRIMARY KEY,
    question_text TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tags (
    tag VARCHAR(255)
);

COMMIT;
