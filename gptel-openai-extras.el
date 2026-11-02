;;; gptel-openai-extras.el --- Extensions to the OpenAI API -*- lexical-binding: t; -*-

;; Copyright (C) 2023-2026  Karthik Chikmagalur

;; Authors: Karthik Chikmagalur <karthikchikmagalur@gmail.com> and pirminj

;; This program is free software; you can redistribute it and/or modify
;; it under the terms of the GNU General Public License as published by
;; the Free Software Foundation, either version 3 of the License, or
;; (at your option) any later version.

;; This program is distributed in the hope that it will be useful,
;; but WITHOUT ANY WARRANTY; without even the implied warranty of
;; MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
;; GNU General Public License for more details.

;; You should have received a copy of the GNU General Public License
;; along with this program.  If not, see <https://www.gnu.org/licenses/>.

;;; Commentary:

;; This file adds support for Privategpt's Messages API,
;; Perplexity's Citations feature, the Deepseek and xAI APIs, and
;; OpenCode Go to gptel

;;; Code:
(require 'cl-generic)
(eval-when-compile (require 'cl-lib))
(require 'map)
(eval-and-compile
  (require 'gptel-request)
  (require 'gptel-openai))

(defvar json-object-type)
(defvar gptel-version)

(declare-function prop-match-value "text-property-search" nil t)
(declare-function text-property-search-backward "text-property-search")
(declare-function json-read "json" ())



;;; Privategpt (Messages API)
(cl-defstruct (gptel-privategpt (:constructor gptel--make-privategpt)
                               (:copier nil)
                               (:include gptel-openai))
  context sources)

(defun gptel--privategpt-parse-sources (response)
  "Extract sources from a PrivateGPT RESPONSE, as a display string."
  (cl-loop with source-alist
           for source across (map-nested-elt response '(:choices 0 :sources))
           for name = (map-nested-elt source '(:document :doc_metadata :file_name))
           for page = (map-nested-elt source '(:document :doc_metadata :page_label))
           do (push page (alist-get name source-alist nil nil #'equal))
           finally return
           (cl-loop for (file-name . file-pages) in source-alist
                    for pages = (delete-dups (delq nil file-pages))
                    if pages
                    collect (format "- %s (page %s)" file-name (mapconcat #'identity pages ", "))
                    into source-items
                    else collect (format "- %s" file-name) into source-items
                    finally return (mapconcat #'identity (cons "\n\nSources:" source-items) "\n"))))

;; FIXME(tool) add tool use
(cl-defmethod gptel-curl--parse-stream ((_backend gptel-privategpt) info)
  "Parse the PrivateGPT response stream with request plist INFO."
  (let* ((content-strs))
    (condition-case nil
        (while (re-search-forward "^data:" nil t)
          (save-match-data
            (if (looking-at " *\\[DONE\\]")
                (when-let* ((sources-string (plist-get info :sources)))
                  (push sources-string content-strs))
              (let ((response (gptel--json-read)))
		(unless (or (plist-get info :sources)
                            (not (gptel-privategpt-sources (plist-get info :backend))))
                  (plist-put info :sources (gptel--privategpt-parse-sources response)))
		(let* ((delta (map-nested-elt response '(:choices 0 :delta)))
		       (content (plist-get delta :content)))
		  (push content content-strs))))))
      (error
       (goto-char (match-beginning 0))))
    (apply #'concat (nreverse content-strs))))

;; FIXME(tool) add tool use
(cl-defmethod gptel--parse-response ((_backend gptel-privategpt) response info)
  "Parse a PrivateGPT RESPONSE and return its text, with sources.
Sources are included in INFO."
  (let ((response-string (map-nested-elt response '(:choices 0 :message :content)))
        (sources-string (and (gptel-privategpt-sources (plist-get info :backend))
                             (gptel--privategpt-parse-sources response))))
    (concat response-string sources-string)))

(cl-defmethod gptel--request-data ((_backend gptel-privategpt) prompts)
  "JSON encode PROMPTS for sending to ChatGPT."
  (let ((prompts-plist
         `(:model ,(gptel--model-name gptel-model)
	   :messages [,@prompts]
	   :use_context ,(or (gptel-privategpt-context gptel-backend) :json-false)
	   :include_sources ,(or (gptel-privategpt-sources gptel-backend) :json-false)
           :stream ,(or gptel-stream :json-false))))
    (when (and gptel-system-prompt
               (not (gptel--model-capable-p 'nosystem)))
      (plist-put prompts-plist :system gptel-system-prompt))
    (when gptel-temperature
      (plist-put prompts-plist :temperature gptel-temperature))
    (when gptel-max-tokens
      (plist-put prompts-plist :max_tokens gptel-max-tokens))
    ;; Merge request params with model and backend params.
    (gptel--merge-plists
     prompts-plist
     (gptel-backend-request-params gptel-backend)
     (gptel--model-request-params  gptel-model))))


;;;###autoload
(cl-defun gptel-make-privategpt
    (name &key curl-args stream key request-params
          (header
           (lambda (_info)
             (when-let* ((key (gptel--get-api-key)))
	       `(("Authorization" . ,(concat "Bearer " key))))))
          (host "localhost:8001")
          (protocol "http")
	  (models '(private-gpt))
          (endpoint "/v1/chat/completions")
          (context t) (sources t))
  "Register a PrivateGPT backend for gptel with NAME.

Keyword arguments:

CURL-ARGS (optional) is a list of additional Curl arguments.

HOST (optional) is the API host, \"localhost:8001\" by default.

MODELS is a list of available model names.

STREAM is a boolean to toggle streaming responses, defaults to
false.

PROTOCOL (optional) specifies the protocol, https by default.

ENDPOINT (optional) is the API endpoint for completions, defaults to
\"/v1/messages\".

HEADER (optional) is for additional headers to send with each
request.  It should be an alist or a function that returns an
alist, like:
 ((\"Content-Type\" . \"application/json\"))

KEY is a variable whose value is the API key, or function that
returns the key.

CONTEXT and SOURCES: if true (the default), use available context
and provide sources used by the model to generate the response.

REQUEST-PARAMS (optional) is a plist of additional HTTP request
parameters (as plist keys) and values supported by the API.  Use
these to set parameters that gptel does not provide user options
for."
  (declare (indent 1))
  (let ((backend (gptel--make-privategpt
                  :curl-args curl-args
                  :name name
                  :host host
                  :header header
                  :key key
                  :models models
                  :protocol protocol
                  :endpoint endpoint
                  :stream stream
                  :request-params request-params
                  :url (if protocol
                           (concat protocol "://" host endpoint)
                         (concat host endpoint))
                  :context context
                  :sources sources)))
    (prog1 backend
      (setf (alist-get name gptel--known-backends
                       nil nil #'equal)
            backend))))


;;; Perplexity
(cl-defstruct (gptel-perplexity (:constructor gptel--make-perplexity)
				(:copier nil)
				(:include gptel-openai)))

(defsubst gptel--perplexity-parse-citations (citations)
  "Format CITATIONS as a display string for Perplexity responses."
  (let ((counter 0))
    (concat "\n\nCitations:\n"
            (mapconcat (lambda (url)
                         (setq counter (1+ counter))
                         (format "[%d] %s" counter url))
                       citations "\n"))))

(cl-defmethod gptel--parse-response ((_backend gptel-perplexity) response _info)
  "Parse Perplexity response RESPONSE."
  (let ((response-string (cl-call-next-method))
        (citations-string (when-let* ((citations (map-elt response :citations)))
			    (gptel--perplexity-parse-citations citations))))
    (concat response-string citations-string)))

(cl-defmethod gptel-curl--parse-stream ((_backend gptel-perplexity) info)
  "Parse a Perplexity API data stream with INFO.

If available, collect citations at the end and include them with
the response."
  (let ((resp (cl-call-next-method)))
    (unless (plist-get info :citations)
      (save-excursion
        (goto-char (point-max))
        (when (search-backward (plist-get info :uuid)
                               (line-beginning-position) t)
          (forward-line 0)
          (when (re-search-backward "^data: " nil t)
            (goto-char (match-end 0))
            (ignore-errors
              (when-let* ((chunk (gptel--json-read))
                          (citations (map-elt chunk :citations)))
                (plist-put info :citations t)
                (setq resp (concat resp (gptel--perplexity-parse-citations
                                         citations)))))))))
    resp))

;;;###autoload
(cl-defun gptel-make-perplexity
    (name &key curl-args stream key
          (header
           (lambda (_info)
             (when-let* ((key (gptel--get-api-key)))
               `(("Authorization" . ,(concat "Bearer " key))
                 ("X-Pplx-Integration" . "gptel")))))
          (host "api.perplexity.ai")
          (protocol "https")
          ;; https://docs.perplexity.ai/guides/model-cards
          (models '(sonar sonar-pro sonar-reasoning sonar-reasoning-pro sonar-deep-research))
          (endpoint "/chat/completions")
          request-params)
  "Register a Perplexity backend for gptel with NAME.

Keyword arguments:

HOST (optional) is the API host, \"api.perplexity.ai\" by default.

MODELS is a list of available model names.

STREAM is a boolean to toggle streaming responses.

PROTOCOL (optional) specifies the protocol, https by default.

ENDPOINT (optional) is the API endpoint for completions.

HEADER (optional) is for additional headers to send with each
request.  It should be an alist or a function that returns an
alist.

KEY is a variable whose value is the API key, or function that
returns the key.

REQUEST-PARAMS (optional) is a plist of additional HTTP request
parameters.

The remaining keyword arguments, including CURL-ARGS, are
optional.  For their meanings see `gptel-make-openai'."
  (declare (indent 1))
  (let ((backend (gptel--make-perplexity
                  :curl-args curl-args
                  :name name
                  :host host
                  :header header
                  :key key
                  :models models
                  :protocol protocol
                  :endpoint endpoint
                  :stream stream
                  :request-params request-params
                  :url (if protocol
                           (concat protocol "://" host endpoint)
                         (concat host endpoint)))))
    (prog1 backend
      (setf (alist-get name gptel--known-backends
                       nil nil #'equal)
            backend))))

;;; Deepseek
(cl-defstruct (gptel-deepseek (:include gptel-openai)
                              (:copier nil)
                              (:constructor gptel--make-deepseek)))

(cl-defmethod gptel--parse-buffer :around ((_backend gptel-deepseek) _max-entries)
  "Merge successive prompts in the prompts list that have the same role.

The Deepseek API requires strictly alternating roles (user/assistant) in messages."
  (let* ((prompts (cl-call-next-method))
         (index prompts))
    (prog1 prompts
      (while index
        (let ((p1 (car index))
              (p2 (cadr index))
              (rest (cdr index)))
          (when (and p2 (equal (plist-get p1 :role)
                               (plist-get p2 :role)))
            ;; Blocks to be merged must both be text blocks
            ;; and not tool calls
            (when-let* ((content1 (plist-get p1 :content))
                        (content2 (plist-get p2 :content)))
              (plist-put
               p1 :content
               (if (vectorp content1)
                   ;; Case [(:type "text" :text "...")]
                   (vconcat content1 [(:type "text" :text "\n")] content2)
                 (concat content1 "\n" content2))) ;all strings
              (setcdr index (cdr rest))))
          (setq index (cdr index)))))))

(cl-defmethod gptel--request-data :around ((_backend gptel-deepseek) _prompts)
  "Modify how structured output JSON schema is specified for Deepseek.

This method wraps the main `gptel--request-data' implementation
to move the JSON schema from the response format to the system
message."
  (let ((prompts-plist (cl-call-next-method)))
    (when gptel--schema
      (let ((response-format
             (prin1-to-string (plist-get prompts-plist :response_format))))
        (plist-put prompts-plist :response_format (list :type "json_object"))
        (cl-callf (lambda (system)
                    (concat system
                            "\n\n<response_format>Required JSON schema for response:\n\n"
                            response-format "\n</response_format>"))
            (plist-get (aref (plist-get prompts-plist :messages) 0) :content))))
    prompts-plist))

;;;###autoload
(cl-defun gptel-make-deepseek
    (name &key curl-args stream key request-params
          (header (lambda (_info)
                    (when-let* ((key (gptel--get-api-key)))
                      `(("Authorization" . ,(concat "Bearer " key))))))
          (host "api.deepseek.com")
          (protocol "https")
          (endpoint "/v1/chat/completions")
          (models '((deepseek-flash
                     :capabilities (media tool-use reasoning url)
                     :mime-types ("image/jpeg" "image/png" "image/gif" "image/webp")
                     :context-window 1000
                     :input-cost 0.15
                     :output-cost 0.6)
                    (deepseek-v4-pro
                     :capabilities (tool-use reasoning)
                     :context-window 1000
                     :input-cost 0.66
                     :output-cost 1.98)
                    (deepseek-v4-flash
                     :description "DEPRECATED: Use deepseek-flash instead."
                     :capabilities (tool-use reasoning)
                     :context-window 1000
                     :input-cost 0.15
                     :output-cost 0.6)
                    (deepseek-v4-flash-vision-exp
                     :description "DEPRECATED: Use deepseek-flash instead."
                     :capabilities (media tool-use reasoning url)
                     :mime-types ("image/jpeg" "image/png" "image/gif" "image/webp")
                     :context-window 1000
                     :input-cost 0.15
                     :output-cost 0.6))))
  "Register a DeepSeek backend for gptel with NAME.

CURL-ARGS, STREAM, KEY, REQUEST-PARAMS, HEADER, HOST, PROTOCOL,
ENDPOINT and MODELS are all optional; for their meanings, see
`gptel-make-openai'."
  (declare (indent 1))
  (let ((backend (gptel--make-deepseek
                  :name name
                  :host host
                  :header header
                  :key key
                  :models (gptel--process-models models)
                  :protocol protocol
                  :endpoint endpoint
                  :stream stream
                  :request-params request-params
                  :curl-args curl-args
                  :url (concat protocol "://" host endpoint))))
    (setf (alist-get name gptel--known-backends nil nil #'equal) backend)
    backend))

;;; xAI
;;;###autoload
(cl-defun gptel-make-xai
    (name &key curl-args stream key request-params
          (header (lambda (_info)
                    (when-let* ((key (gptel--get-api-key)))
                      `(("Authorization" . ,(concat "Bearer " key))))))
          (host "api.x.ai")
          (protocol "https")
          (endpoint "/v1/chat/completions")
          (models
           '((grok-4.7
              :description "Frontier model for coding, agentic tasks, and knowledge work. The most capable Grok model."
              :capabilities (tool-use json media reasoning)
              :mime-types ("image/jpeg" "image/png")
              :context-window 500
              :input-cost 2
              :output-cost 6)

             (grok-4.6
              :description "Previous frontier model for coding, agentic tasks, and knowledge work."
              :capabilities (tool-use json media reasoning)
              :mime-types ("image/jpeg" "image/png")
              :context-window 500
              :input-cost 2
              :output-cost 6)

             (grok-4.5
              :description "Coding model for agentic software, engineering, and workflow tasks."
              :capabilities (tool-use json media reasoning)
              :mime-types ("image/jpeg" "image/png" "image/gif" "image/webp")
              :context-window 500
              :input-cost 2
              :output-cost 6)

             (grok-4.3
              :description "Fast, lower-cost model with strong tool calling and instruction following. 1M context."
              :capabilities (tool-use json media reasoning)
              :mime-types ("image/jpeg" "image/png" "image/gif" "image/webp")
              :context-window 1000
              :input-cost 1.25
              :output-cost 2.5)

             (grok-4.20
              :description "Fast agentic model with strict prompt adherence."
              :capabilities (tool-use json media reasoning)
              :mime-types ("image/jpeg" "image/png")
              :context-window 1000
              :input-cost 1.25
              :output-cost 2.5)

             (grok-4.20-non-reasoning
              :description "Non-reasoning variant of Grok 4.20, optimized for speed and prompt adherence."
              :capabilities (tool-use json media)
              :mime-types ("image/jpeg" "image/png")
              :context-window 1000
              :input-cost 1.25
              :output-cost 2.5)

             (grok-4.20-multi-agent
              :description "Multi-agent variant of Grok 4.20; agents collaborate in parallel on deep research."
              :capabilities (tool-use json media reasoning)
              :mime-types ("image/jpeg" "image/png")
              :context-window 1000
              :input-cost 1.25
              :output-cost 2.5)

             (grok-build-0.1
              :description "Coding model for agentic software, engineering, and workflow tasks. 256k context."
              :capabilities (tool-use json media reasoning)
              :mime-types ("image/jpeg" "image/png" "image/gif" "image/webp")
              :context-window 256
              :input-cost 1
              :output-cost 2))))
  "Register an xAI backend for gptel with NAME.

Keyword arguments:

KEY is a variable whose value is the API key, or function that
returns the key.

STREAM is a boolean to toggle streaming responses, defaults to
false.

The remaining keyword arguments (CURL-ARGS, REQUEST-PARAMS,
HEADER, HOST, PROTOCOL, ENDPOINT and MODELS) are optional; for
their meanings see `gptel-make-openai'."
  (declare (indent 1))
  (let ((backend (gptel--make-deepseek
                  :name name
                  :host host
                  :header header
                  :key key
                  :models (gptel--process-models models)
                  :protocol protocol
                  :endpoint endpoint
                  :stream stream
                  :request-params request-params
                  :curl-args curl-args
                  :url (concat protocol "://" host endpoint))))
    (setf (alist-get name gptel--known-backends nil nil #'equal) backend)
    backend))


;;; OpenCode Go
(cl-defstruct (gptel-opencode (:include gptel-openai)
                              (:copier nil)
                              (:constructor gptel--make-opencode)))

(defvar-local gptel-opencode-session-id nil
  "Session ID used for OpenCode Go requests from the current buffer.

OpenCode Go requires every request to carry a stable identifier
for its conversation in the \"x-opencode-session\" header, which
it uses for request routing and prompt caching.  The ID is
generated on the first request from a buffer and reused for all
subsequent requests from it.  Set this variable to override the
generated value.")

(defun gptel--opencode-buffer-session (buffer)
  "Return the OpenCode session ID for BUFFER, generating it if needed."
  (with-current-buffer buffer
    (or gptel-opencode-session-id
        (setq gptel-opencode-session-id
              (format "gptel-%s"
                      (md5 (format "%s-%s-%s"
                                   (buffer-name)
                                   (float-time)
                                   (random))))))))

(defun gptel--opencode-header (info)
  "Return the HTTP headers for an OpenCode Go request.

INFO is the gptel request plist.  Besides the API key, send the
client identification and per-conversation session headers that
OpenCode Go requires."
  (append
   (when-let* ((key (gptel--get-api-key)))
     `(("Authorization" . ,(concat "Bearer " key))))
   `(("User-Agent" . ,(format "gptel/%s"
                              (if (boundp 'gptel-version) gptel-version "unknown")))
     ("x-opencode-session" . ,(gptel--opencode-buffer-session
                               (or (plist-get info :buffer)
                                   (current-buffer)))))))

(defconst gptel--opencode-models
  '((deepseek-v4.1-flash
     :description "DeepSeek V4.1 Flash model for reasoning and agentic coding"
     :capabilities (media tool-use json reasoning)
     :mime-types ("image/jpeg" "image/png" "image/gif" "image/webp")
     :context-window 1000
     :input-cost 0.15
     :output-cost 0.6)
    (deepseek-v4-pro
     :description "Flagship DeepSeek model for coding, reasoning, and agentic work"
     :capabilities (tool-use json reasoning)
     :context-window 1000
     :input-cost 0.66
     :output-cost 1.98)
    (glm-5.3-flash
     :description "Native multimodal GLM model for efficient coding and long-horizon agent tasks"
     :capabilities (media tool-use json reasoning)
     :mime-types ("image/jpeg" "image/png")
     :context-window 1000
     :input-cost 0.15
     :output-cost 0.5)
    (glm-5.3
     :description "Flagship GLM model for long-horizon coding, agents, and complex project delivery"
     :capabilities (tool-use json reasoning)
     :context-window 1000
     :input-cost 1.4
     :output-cost 4.4)
    (glm-5.2
     :description "Open flagship GLM for long-horizon coding agents and million-token context work"
     :capabilities (tool-use json reasoning)
     :context-window 1000
     :input-cost 1.4
     :output-cost 4.4)
    (kimi-k3.1-pro
     :description "Multimodal Kimi model with 1M context and always-on reasoning for coding, 3D creation, financial research, and agentic work"
     :capabilities (media tool-use json reasoning)
     :mime-types
     ("image/jpeg" "image/png" "image/gif" "image/webp" "image/bmp" "image/heic" "image/heif")
     :context-window 1024
     :input-cost 2
     :output-cost 10)
    (kimi-k3
     :description "Multimodal Kimi model with 1M context and toggleable max-effort thinking for long-horizon agent work"
     :capabilities (media tool-use json reasoning)
     :mime-types
     ("image/jpeg" "image/png" "image/gif" "image/webp" "image/bmp" "image/heic" "image/heif")
     :context-window 1024
     :input-cost 3
     :output-cost 15)
    (kimi-k2.7-code
     :description "Coding-focused Kimi model, stronger on long-horizon repo work with less overthinking"
     :capabilities (media tool-use json reasoning)
     :mime-types
     ("image/jpeg" "image/png" "image/gif" "image/webp" "image/bmp" "image/heic" "image/heif")
     :context-window 262
     :input-cost 0.95
     :output-cost 4)
    (qwen3.8-max
     :description "2.4-trillion-parameter multimodal flagship for coding, professional work, and long-horizon agentic workflows"
     :capabilities (media tool-use json reasoning)
     :mime-types
     ("image/bmp" "image/jpeg" "image/png" "image/tiff" "image/webp" "image/heic")
     :context-window 1000
     :input-cost 2
     :output-cost 6)
    (qwen3.8-flash
     :description "Qwen vision-language model for visual reasoning, documents, and agent tasks"
     :capabilities (media tool-use json reasoning)
     :mime-types
     ("image/bmp" "image/jpeg" "image/png" "image/tiff" "image/webp" "image/heic")
     :context-window 1000
     :input-cost 0.15
     :output-cost 0.47)
    (qwen3.7-plus
     :description "Multimodal reasoning model for visual analysis, planning, and tool use"
     :capabilities (media tool-use reasoning)
     :mime-types
     ("image/bmp" "image/jpeg" "image/png" "image/tiff" "image/webp" "image/heic")
     :context-window 1000
     :input-cost 0.4
     :output-cost 1.6)
    (minimax-m3.1-flash
     :description "Frontier multimodal coding model with 1M context window and tunable thinking depth"
     :capabilities (media tool-use reasoning)
     :mime-types ("image/jpeg" "image/png" "image/gif" "image/webp")
     :context-window 1000
     :input-cost 0.15
     :output-cost 0.6)
    (minimax-m3
     :description "MiniMax multimodal coding model for long-context reasoning and agent tasks"
     :capabilities (media tool-use reasoning)
     :mime-types ("image/jpeg" "image/png" "image/gif" "image/webp")
     :context-window 1000
     :input-cost 0.3
     :output-cost 1.2)
    (mimo-v2.6-pro
     :description "Stronger MiMo Pro tier for multimodal reasoning and coding-agent execution"
     :capabilities (media tool-use reasoning)
     :mime-types
     ("image/jpeg" "image/png" "image/gif" "image/webp" "image/bmp")
     :context-window 1024
     :input-cost 0.435
     :output-cost 0.87)
    (mimo-v2.6-flash
     :description "MiMo Flash model for multimodal coding agents and long-context automation"
     :capabilities (media tool-use reasoning)
     :mime-types
     ("image/jpeg" "image/png" "image/gif" "image/webp" "image/bmp")
     :context-window 1024
     :input-cost 0.14
     :output-cost 0.28)
    (grok-4.7
     :description "xAI's frontier model for long-running agents, coding, knowledge work, and visual projects"
     :capabilities (media tool-use json reasoning)
     :mime-types ("image/jpeg" "image/png" "application/pdf")
     :context-window 500
     :input-cost 2
     :output-cost 6)
    (gpt-6-luna
     :description "OpenAI's most efficient model for focused, high-volume tasks"
     :capabilities (media tool-use json reasoning)
     :mime-types
     ("image/jpeg" "image/png" "image/gif" "image/webp" "application/pdf")
     :context-window 1050
     :input-cost 0.1
     :output-cost 0.5)
    (gpt-5.6-luna
     :description "Cost-efficient GPT-5.6 model for fast, high-volume workloads"
     :capabilities (media tool-use json reasoning)
     :mime-types
     ("image/jpeg" "image/png" "image/gif" "image/webp" "application/pdf")
     :context-window 1050
     :input-cost 0.2
     :output-cost 1.2)
    (claude-haiku-5-5
     :description "Fast Claude model for responsive assistance, classification, and lightweight agents"
     :capabilities (media tool-use json reasoning)
     :mime-types
     ("image/jpeg" "image/png" "image/gif" "image/webp" "application/pdf")
     :context-window 1000
     :input-cost 0.1
     :output-cost 0.5)
    (gemini-4-argon
     :description "Google's flagship reasoning model for complex software engineering, deep math and science, and long-horizon autonomous agents"
     :capabilities (media tool-use json reasoning)
     :mime-types
     ("image/png" "image/jpeg" "image/webp" "image/heic" "image/heif" "application/pdf")
     :context-window 1024
     :input-cost 2
     :output-cost 10)
    (hy4-preview
     :description "A next-generation productivity model with significantly enhanced Agent and complex task execution capabilities."
     :capabilities (tool-use reasoning)
     :context-window 1024
     :input-cost 0.834
     :output-cost 2.501)
    (hy3
     :description "Tencent Hy reasoning model for coding, instruction following, and agent tasks"
     :capabilities (tool-use reasoning)
     :context-window 256
     :input-cost 0.14
     :output-cost 0.58)
    (longcat-2.0
     :description "Meituan LongCat-2.0, a reasoning model with tool calling and a 1M-token context window"
     :capabilities (tool-use reasoning)
     :context-window 1000
     :input-cost 0.3
     :output-cost 1.2)
    (step-5-preview
     :description "StepFun's next-generation flagship base model for coding and professional knowledge work, with native text, image, and video input and a 1M-token context window"
     :capabilities (media tool-use json reasoning)
     :mime-types ("image/jpeg" "image/png" "image/gif" "image/webp")
     :context-window 1000
     :input-cost 1
     :output-cost 2.7)
    (muse-spark-1.3-contributor
     :description "Muse Spark 1.3 is a multimodal reasoning model from Meta for coding and agentic workflows."
     :capabilities (media tool-use json reasoning)
     :mime-types
     ("image/jpeg" "image/png" "image/gif" "image/webp" "application/pdf")
     :context-window 1024
     :input-cost 0.1
     :output-cost 0.2)
    (muse-spark-1.2-contributor
     :description "Muse Spark 1.2 is a coding-focused update to Muse Spark 1.1 with improvements in code generation, complex debugging, codebase understanding, and end-to-end developer workflows."
     :capabilities (media tool-use json reasoning)
     :mime-types
     ("image/jpeg" "image/png" "image/gif" "image/webp" "application/pdf")
     :context-window 1024
     :input-cost 0.1
     :output-cost 0.2)
    (beam-free
     :description "Sparse mixture-of-experts reasoning model with 23B active parameters for coding and agentic workflows"
     :capabilities (tool-use json reasoning)
     :context-window 1024
     :input-cost 0
     :output-cost 0)
    (iceman-contributor-free
     :description "Stealth model, free for a limited time"
     :capabilities (tool-use reasoning)
     :context-window 262
     :input-cost 0
     :output-cost 0))
  "List of models available through OpenCode Go.

Each model symbol is associated with the following keys, all
optional:

- `:description': a brief description of the model.

- `:capabilities': a list of capabilities supported by the model.

- `:mime-types': a list of supported MIME types for media files.

- `:context-window': the context window size, in thousands of tokens.

- `:input-cost': the input cost, in US dollars per million tokens.

- `:output-cost': the output cost, in US dollars per million tokens.

The OpenCode Go model list changes frequently.  Information about
the models was obtained from the API:

- <https://opencode.ai/docs/go>")

;;;###autoload
(cl-defun gptel-make-opencode
    (name &key curl-args stream key request-params
          (header #'gptel--opencode-header)
          (host "opencode.ai")
          (protocol "https")
          (endpoint "/inference/go/openai/v1/chat/completions")
          (models gptel--opencode-models))
  "Register an OpenCode Go backend for gptel with NAME.

OpenCode Go is a subscription plan for coding models served by
OpenCode at <https://opencode.ai/docs/go>.  Requests are sent to
the OpenAI-compatible endpoint, with the additional headers that
OpenCode Go requires (see `gptel--opencode-header').

Keyword arguments:

KEY is a variable whose value is the API key, or function that
returns the key.

STREAM is a boolean to toggle streaming responses, defaults to
false.

The remaining keyword arguments (CURL-ARGS, REQUEST-PARAMS,
HEADER, HOST, PROTOCOL, ENDPOINT and MODELS) are optional; for
their meanings see `gptel-make-openai'."
  (declare (indent 1))
  (let ((backend (gptel--make-opencode
                  :name name
                  :host host
                  :header header
                  :key key
                  :models (gptel--process-models models)
                  :protocol protocol
                  :endpoint endpoint
                  :stream stream
                  :request-params request-params
                  :curl-args curl-args
                  :url (concat protocol "://" host endpoint))))
    (setf (alist-get name gptel--known-backends nil nil #'equal) backend)
    backend))


(provide 'gptel-openai-extras)
;;; gptel-openai-extras.el ends here

;; Local Variables:
;; byte-compile-warnings: (not docstrings)
;; End:
