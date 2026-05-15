--
-- PostgreSQL database dump
--

\restrict u3Qi2KSPmGRCv82EHHfed378we42ygX6tM6UvfwNvqKrCQE4iEcGFFa1rtca8FV

-- Dumped from database version 17.9
-- Dumped by pg_dump version 17.9

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: vector; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public;


--
-- Name: EXTENSION vector; Type: COMMENT; Schema: -; Owner: 
--

COMMENT ON EXTENSION vector IS 'vector data type and ivfflat and hnsw access methods';


--
-- Name: set_updated_at(); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.set_updated_at() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;


ALTER FUNCTION public.set_updated_at() OWNER TO postgres;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: chunks; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.chunks (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    chunk_text text NOT NULL,
    embedding public.vector(1024),
    category_legacy text NOT NULL,
    source_type text NOT NULL,
    source text NOT NULL,
    law_name text,
    article_number text,
    nn_reference text,
    valid_from date NOT NULL,
    valid_to date,
    status text NOT NULL,
    citable boolean DEFAULT true NOT NULL,
    chunk_index integer,
    total_chunks integer,
    extra_metadata jsonb DEFAULT '{}'::jsonb NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL,
    domain text,
    subdomain text
);


ALTER TABLE public.chunks OWNER TO postgres;

--
-- Name: feedback; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.feedback (
    feedback_id uuid DEFAULT gen_random_uuid() NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    query_id uuid NOT NULL,
    advisor_id text NOT NULL,
    rating smallint NOT NULL,
    accuracy_verdict text,
    would_send_to_client text,
    failure_mode text,
    comment text,
    suggested_answer text
);


ALTER TABLE public.feedback OWNER TO postgres;

--
-- Name: queries; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.queries (
    query_id uuid DEFAULT gen_random_uuid() NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    advisor_id text NOT NULL,
    question_text text NOT NULL,
    classified_category text,
    classifier_confidence double precision,
    retrieved_chunk_ids text[],
    retrieved_scores double precision[],
    answer_text text,
    citations jsonb,
    confidence text,
    referred_to_advisor boolean,
    model_used text,
    tokens_in integer,
    tokens_out integer,
    latency_ms integer,
    estimated_cost_usd double precision,
    error_flag boolean DEFAULT false,
    trace_json jsonb
);


ALTER TABLE public.queries OWNER TO postgres;

--
-- Name: chunks chunks_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.chunks
    ADD CONSTRAINT chunks_pkey PRIMARY KEY (id);


--
-- Name: feedback feedback_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.feedback
    ADD CONSTRAINT feedback_pkey PRIMARY KEY (feedback_id);


--
-- Name: queries queries_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.queries
    ADD CONSTRAINT queries_pkey PRIMARY KEY (query_id);


--
-- Name: feedback_accuracy_verdict_idx; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX feedback_accuracy_verdict_idx ON public.feedback USING btree (accuracy_verdict);


--
-- Name: feedback_query_id_idx; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX feedback_query_id_idx ON public.feedback USING btree (query_id);


--
-- Name: idx_chunks_cat_status; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_chunks_cat_status ON public.chunks USING btree (category_legacy, status);


--
-- Name: idx_chunks_category; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_chunks_category ON public.chunks USING btree (category_legacy);


--
-- Name: idx_chunks_chunk_text_fts; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_chunks_chunk_text_fts ON public.chunks USING gin (to_tsvector('simple'::regconfig, chunk_text));


--
-- Name: idx_chunks_citable; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_chunks_citable ON public.chunks USING btree (citable);


--
-- Name: idx_chunks_domain; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_chunks_domain ON public.chunks USING btree (domain);


--
-- Name: idx_chunks_domain_subdomain; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_chunks_domain_subdomain ON public.chunks USING btree (domain, subdomain);


--
-- Name: idx_chunks_domain_subdomain_status; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_chunks_domain_subdomain_status ON public.chunks USING btree (domain, subdomain, status);


--
-- Name: idx_chunks_embedding_hnsw; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_chunks_embedding_hnsw ON public.chunks USING hnsw (embedding public.vector_cosine_ops) WITH (m='16', ef_construction='64');


--
-- Name: idx_chunks_law_article; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_chunks_law_article ON public.chunks USING btree (law_name, article_number);


--
-- Name: idx_chunks_source_type; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_chunks_source_type ON public.chunks USING btree (source_type);


--
-- Name: idx_chunks_status; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_chunks_status ON public.chunks USING btree (status);


--
-- Name: idx_chunks_subdomain; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_chunks_subdomain ON public.chunks USING btree (subdomain);


--
-- Name: idx_chunks_valid_from; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_chunks_valid_from ON public.chunks USING btree (valid_from);


--
-- Name: idx_chunks_valid_to; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_chunks_valid_to ON public.chunks USING btree (valid_to);


--
-- Name: queries_advisor_id_idx; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX queries_advisor_id_idx ON public.queries USING btree (advisor_id);


--
-- Name: queries_classified_category_idx; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX queries_classified_category_idx ON public.queries USING btree (classified_category);


--
-- Name: queries_created_at_idx; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX queries_created_at_idx ON public.queries USING btree (created_at DESC);


--
-- Name: chunks trg_chunks_updated_at; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER trg_chunks_updated_at BEFORE UPDATE ON public.chunks FOR EACH ROW EXECUTE FUNCTION public.set_updated_at();


--
-- Name: feedback feedback_query_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.feedback
    ADD CONSTRAINT feedback_query_id_fkey FOREIGN KEY (query_id) REFERENCES public.queries(query_id);


--
-- PostgreSQL database dump complete
--

\unrestrict u3Qi2KSPmGRCv82EHHfed378we42ygX6tM6UvfwNvqKrCQE4iEcGFFa1rtca8FV

