BEGIN;

CREATE TABLE IF NOT EXISTS activities
(
    activity_id integer NOT NULL DEFAULT nextval('activities_activity_id_seq'::regclass),
    activity_text character varying(255) COLLATE pg_catalog."default" NOT NULL,
    icon_url text COLLATE pg_catalog."default",
    CONSTRAINT activities_pkey PRIMARY KEY (activity_id)
);

CREATE TABLE IF NOT EXISTS activity_recommendations
(
    recommendation_id integer NOT NULL DEFAULT nextval('activity_recommendations_recommendation_id_seq'::regclass),
    type character varying(50) COLLATE pg_catalog."default",
    activity_name character varying(255) COLLATE pg_catalog."default",
    CONSTRAINT activity_recommendations_pkey PRIMARY KEY (recommendation_id),
    CONSTRAINT activity_recommendations_activity_name_key UNIQUE (activity_name)
);

CREATE TABLE IF NOT EXISTS cbt_questions
(
    question_id integer NOT NULL DEFAULT nextval('cbt_questions_question_id_seq'::regclass),
    type_id integer,
    question_text text COLLATE pg_catalog."default" NOT NULL,
    CONSTRAINT cbt_questions_pkey PRIMARY KEY (question_id),
    CONSTRAINT fk_cbt_types FOREIGN KEY (type_id) REFERENCES cbt_types(type_id)
);

CREATE TABLE IF NOT EXISTS cbt_types
(
    type_id integer NOT NULL DEFAULT nextval('cbt_types_type_id_seq'::regclass),
    type_name character varying(255) COLLATE pg_catalog."default" NOT NULL,
    icon_url text COLLATE pg_catalog."default",
    type_info text COLLATE pg_catalog."default",
    CONSTRAINT cbt_types_pkey PRIMARY KEY (type_id)
);

CREATE TABLE IF NOT EXISTS daily_emotionStress_tracks
(
    track_id integer NOT NULL DEFAULT nextval('daily_emotion_tracks_track_id_seq'::regclass),
    user_id integer,
    emotion_id integer,
    sub_emotion_id integer,
    has_stress boolean,
    track_date timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    additional_notes text,
    CONSTRAINT daily_emotion_tracks_pkey PRIMARY KEY (track_id),
    CONSTRAINT fk_emotion FOREIGN KEY (emotion_id) REFERENCES emotions(emotion_id),
    CONSTRAINT fk_sub_emotion FOREIGN KEY (sub_emotion_id) REFERENCES sub_emotions(sub_emotion_id),
  CONSTRAINT daily_emotionStress_tracks_user_id_fkey FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE TABLE IF NOT EXISTS daily_activity_tracks
(
    track_id integer NOT NULL DEFAULT nextval('daily_emotion_tracks_track_id_seq'::regclass),
    user_id integer,
    reason_id integer,
    activity_id integer,
    CONSTRAINT daily_activity_tracks_pkey PRIMARY KEY (track_id),
    CONSTRAINT fk_reason FOREIGN KEY (reason_id) REFERENCES reasons(reason_id),
    CONSTRAINT fk_activity FOREIGN KEY (activity_id) REFERENCES activities(activity_id),
    CONSTRAINT daily_activity_tracks_user_id_fkey FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE TABLE IF NOT EXISTS depression_test_scores
(
    score_id integer NOT NULL DEFAULT nextval('depression_test_scores_score_id_seq'::regclass),
    user_id integer,
    score integer NOT NULL,
    test_date timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT depression_test_scores_pkey PRIMARY KEY (score_id),
  CONSTRAINT user_Metadata_user_id_fkey FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE TABLE IF NOT EXISTS depression_test_testquestion
(
    id bigint NOT NULL GENERATED BY DEFAULT AS IDENTITY ( INCREMENT 1 START 1 MINVALUE 1 MAXVALUE 9223372036854775807 CACHE 1 ),
    question character varying(255) COLLATE pg_catalog."default" NOT NULL,
    CONSTRAINT depression_test_testquestion_pkey PRIMARY KEY (id)
);

CREATE TABLE IF NOT EXISTS depression_test_testquestion_answer_options
(
    id bigint NOT NULL GENERATED BY DEFAULT AS IDENTITY ( INCREMENT 1 START 1 MINVALUE 1 MAXVALUE 9223372036854775807 CACHE 1 ),
    testquestion_id bigint NOT NULL,
    answeroption_id bigint NOT NULL,
    CONSTRAINT depression_test_testquestion_answer_options_pkey PRIMARY KEY (id),
    CONSTRAINT depression_test_testques_testquestion_id_answerop_020f5d71_uniq UNIQUE (testquestion_id, answeroption_id),
    CONSTRAINT fk_testquestion FOREIGN KEY (testquestion_id) REFERENCES depression_test_testquestion(id),
    CONSTRAINT fk_answeroption FOREIGN KEY (answeroption_id) REFERENCES depression_test_answeroption(id)
);

CREATE TABLE IF NOT EXISTS depression_test_answeroption
(
    id bigint NOT NULL GENERATED BY DEFAULT AS IDENTITY ( INCREMENT 1 START 1 MINVALUE 1 MAXVALUE 9223372036854775807 CACHE 1 ),
    value integer NOT NULL,
    label character varying(255) COLLATE pg_catalog."default" NOT NULL,
    CONSTRAINT depression_test_answeroption_pkey PRIMARY KEY (id)
);

CREATE TABLE IF NOT EXISTS emotions
(
    emotion_id integer NOT NULL DEFAULT nextval('emotions_emotion_id_seq'::regclass),
    emotion_name character varying(255) COLLATE pg_catalog."default" NOT NULL,
    icon_url text COLLATE pg_catalog."default",
    CONSTRAINT emotions_pkey PRIMARY KEY (emotion_id)
);




CREATE TABLE IF NOT EXISTS userplan_activities
(
    user_activity_id integer NOT NULL DEFAULT nextval('userplan_activities_user_activity_id_seq'::regclass),
    user_id integer,
    activity_id integer,
    CONSTRAINT userplan_activities_pkey PRIMARY KEY (user_activity_id),
    CONSTRAINT fk_userplan_activities_user_id FOREIGN KEY (user_id) REFERENCES users(user_id),
    CONSTRAINT fk_userplan_activities_activity_id FOREIGN KEY (activity_id) REFERENCES plan_activities(activity_id)
);

CREATE TABLE IF NOT EXISTS plan_activities
(
    activity_id integer NOT NULL DEFAULT nextval('plan_activities_activity_id_seq'::regclass),
    topic_id integer,
    activity_description text COLLATE pg_catalog."default" NOT NULL,
    CONSTRAINT plan_activities_pkey PRIMARY KEY (activity_id),
    CONSTRAINT fk_plan_activities_topic_id FOREIGN KEY (topic_id) REFERENCES topics(topic_id)
);

CREATE TABLE IF NOT EXISTS preferences_questions
(
    question_id integer NOT NULL DEFAULT nextval('preferences_questions_question_id_seq'::regclass),
    question_text text COLLATE pg_catalog."default" NOT NULL,
    tag VARCHAR(255),
    CONSTRAINT preferences_questions_pkey PRIMARY KEY (question_id),
    CONSTRAINT fk_tag
        FOREIGN KEY (tag)
        REFERENCES tags (tag)
);
CREATE TABLE IF NOT EXISTS user_preferences_answers
(
    answer_id integer NOT NULL DEFAULT nextval('user_preferences_answers_answer_id_seq'::regclass),
    user_id integer,
    tag VARCHAR(255),
    answer boolean NOT NULL,
    CONSTRAINT user_preferences_answers_pkey PRIMARY KEY (answer_id),
    CONSTRAINT fk_user_preferences_answers_user_id FOREIGN KEY (user_id) REFERENCES users(user_id),
    CONSTRAINT fk_user_preferences_answers_tag FOREIGN KEY (tag) REFERENCES preferences_questions(tag)
);
CREATE TABLE IF NOT EXISTS reasons
(
    reason_id integer NOT NULL DEFAULT nextval('reasons_reason_id_seq'::regclass),
    reason_text character varying(255) COLLATE pg_catalog."default" NOT NULL,
    icon_url text COLLATE pg_catalog."default",
    CONSTRAINT reasons_pkey PRIMARY KEY (reason_id)
);

CREATE TABLE IF NOT EXISTS sub_emotions
(
    sub_emotion_id integer NOT NULL DEFAULT nextval('sub_emotions_sub_emotion_id_seq'::regclass),
    emotion_id integer,
    sub_emotion_name character varying(255) COLLATE pg_catalog."default" NOT NULL,
    icon_url text COLLATE pg_catalog."default",
    description text COLLATE pg_catalog."default",
    tips text COLLATE pg_catalog."default",
    CONSTRAINT sub_emotions_pkey PRIMARY KEY (sub_emotion_id)
);

CREATE TABLE IF NOT EXISTS topics
(
    topic_id integer NOT NULL DEFAULT nextval('topics_topic_id_seq'::regclass),
    topic_name character varying(255) COLLATE pg_catalog."default" NOT NULL,
    CONSTRAINT topics_pkey PRIMARY KEY (topic_id)
);

CREATE TABLE IF NOT EXISTS plan_activities
(
    activity_id integer NOT NULL DEFAULT nextval('plan_activities_activity_id_seq'::regclass),
    topic_id integer,
    activity_description text COLLATE pg_catalog."default" NOT NULL,
    CONSTRAINT plan_activities_pkey PRIMARY KEY (activity_id),
    CONSTRAINT fk_plan_activities_topic_id FOREIGN KEY (topic_id) REFERENCES topics(topic_id)
);
CREATE TABLE IF NOT EXISTS learning_topics
(
    learning_topic_id integer NOT NULL DEFAULT nextval('learning_topics_learning_topic_id_seq'::regclass),
    topic_name character varying(255) COLLATE pg_catalog."default" NOT NULL,
    CONSTRAINT learning_topics_pkey PRIMARY KEY (learning_topic_id)
);

CREATE TABLE IF NOT EXISTS lessons
(
    lesson_id integer NOT NULL DEFAULT nextval('lessons_lesson_id_seq'::regclass),
    learning_topic_id integer,
    lesson_title character varying(255) COLLATE pg_catalog."default" NOT NULL,
    lesson_content text COLLATE pg_catalog."default" NOT NULL,
    CONSTRAINT lessons_pkey PRIMARY KEY (lesson_id),
    CONSTRAINT fk_lessons_learning_topic_id FOREIGN KEY (learning_topic_id) REFERENCES learning_topics(learning_topic_id)
);


CREATE TABLE IF NOT EXISTS users
(
    user_id integer NOT NULL DEFAULT nextval('users_user_id_seq'::regclass),
    email character varying(255) COLLATE pg_catalog."default" NOT NULL,
    password character varying(255) COLLATE pg_catalog."default" NOT NULL,
    CONSTRAINT users_pkey PRIMARY KEY (user_id),
    CONSTRAINT users_email_key UNIQUE (email)
);

CREATE TABLE IF NOT EXISTS user_Metadata
(
    setting_id integer NOT NULL DEFAULT nextval('user_settings_setting_id_seq'::regclass),
    user_id integer UNIQUE,
    notifications_enabled boolean DEFAULT true,
    start_day_time time without time zone,
    image_url text COLLATE pg_catalog."default",
    date_of_birth timestamp with time zone,
    gender character varying(50) COLLATE pg_catalog."default",
    first_name character varying(100) COLLATE pg_catalog."default",
    last_name character varying(100) COLLATE pg_catalog."default",
    CONSTRAINT user_Metadata_pkey PRIMARY KEY (setting_id),
    CONSTRAINT user_Metadata_user_id_fkey FOREIGN KEY (user_id) REFERENCES users(user_id),
    CONSTRAINT user_Metadata_email_key UNIQUE (email)
);

CREATE TABLE IF NOT EXISTS user_topics
(
    user_topic_id integer NOT NULL DEFAULT nextval('user_topics_user_topic_id_seq'::regclass),
    user_id integer,
    topic_id integer,
    CONSTRAINT user_topics_pkey PRIMARY KEY (user_topic_id),
    CONSTRAINT fk_user_topics_user_id FOREIGN KEY (user_id) REFERENCES users(user_id),
    CONSTRAINT fk_user_topics_topic_id FOREIGN KEY (topic_id) REFERENCES topics(topic_id)
);
CREATE TABLE IF NOT EXISTS topics
(
    topic_id integer NOT NULL DEFAULT nextval('topics_topic_id_seq'::regclass),
    topic_name character varying(255) COLLATE pg_catalog."default" NOT NULL,
    CONSTRAINT topics_pkey PRIMARY KEY (topic_id)
);
CREATE TABLE IF NOT EXISTS user_weekly_answers
(
    answer_id integer NOT NULL DEFAULT nextval('user_weekly_answers_answer_id_seq'::regclass),
    user_id integer,
    question_id integer,
    answer integer,
    answer_date timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT user_weekly_answers_pkey PRIMARY KEY (answer_id),
    CONSTRAINT fk_user_weekly_answers_user_id FOREIGN KEY (user_id) REFERENCES users(user_id),
    CONSTRAINT fk_user_weekly_answers_question_id FOREIGN KEY (question_id) REFERENCES weekly_checking_questions(question_id)
);


CREATE TABLE IF NOT EXISTS weekly_checking_questions
(
    question_id integer NOT NULL DEFAULT nextval('weekly_checking_questions_question_id_seq'::regclass),
    question_text text COLLATE pg_catalog."default" NOT NULL,
    CONSTRAINT weekly_checking_questions_pkey PRIMARY KEY (question_id)
);

CREATE TABLE tags (
    tag VARCHAR(255)
);

COMMIT;