"""
Comprehensive tests for alm.ingestion.transformations module.

Tests cover all functions with edge cases, boundary conditions, and informative
assertions to help diagnose issues quickly.
"""

import pytest
from alm.ingestion.transformations import (
    select_error_logs,
    detect_error_level,
    extract_error_from_log,
    slice_log_message,
    filter_ingoring,
    pre_proccess_log,
    pre_proccess_log_without_extraction,
    proccess_log_inference,
    clean_slash,
)
from alm.models import DetectedLevel


# =============================================================================
# Test Fixtures - Sample Log Data
# =============================================================================


@pytest.fixture
def fatal_log_line():
    """Standard fatal log line with proper format."""
    return (
        'fatal: [host1.example.com]: FAILED! => {"msg": "Connection timeout", "rc": 1}'
    )


@pytest.fixture
def error_log_line():
    """Standard error log line with proper format."""
    return (
        'error: [host2.example.com]: UNREACHABLE! => {"msg": "SSH connection refused"}'
    )


@pytest.fixture
def failed_log_line():
    """Standard failed log line with proper format."""
    return 'failed: [host3.example.com] (item=package1) => {"msg": "Package not found", "rc": 127}'


@pytest.fixture
def ignoring_log_line():
    """Log line that should be filtered due to ...ignoring suffix."""
    return 'fatal: [host1.example.com]: FAILED! => {"msg": "Error"} ...ignoring'


@pytest.fixture
def success_log_line():
    """Successful log line that should not be selected as error."""
    return 'ok: [host1.example.com] => {"changed": false, "msg": "OK"}'


@pytest.fixture
def multi_line_log_sample():
    """Multi-line log sample with mixed content."""
    return """TASK [Setup package manager] ************************************

ok: [host1.example.com]

fatal: [host2.example.com]: FAILED! => {"msg": "Package manager not found", "rc": 1}

error: [host3.example.com]: UNREACHABLE! => {"msg": "Connection lost"}

failed: [host4.example.com] (item=nginx) => {"msg": "Failed to install nginx"}

ok: [host5.example.com] => {"changed": true}

fatal: [host6.example.com]: FAILED! => {"msg": "Timeout"} ...ignoring"""


# =============================================================================
# Tests for select_error_logs()
# =============================================================================


class TestSelectErrorLogs:
    """Tests for select_error_logs function - filters error logs from multi-line text."""

    def test_selects_fatal_logs(self, multi_line_log_sample):
        """GIVEN multi-line logs with fatal entries
        WHEN select_error_logs is called
        THEN it should include fatal logs (except those with ...ignoring)."""
        result = select_error_logs(multi_line_log_sample)

        # Should find fatal log without ...ignoring
        fatal_lines = [line for line in result if "fatal:" in line]
        assert len(fatal_lines) == 1, (
            f"Expected exactly 1 fatal log (excluding ...ignoring), "
            f"got {len(fatal_lines)}: {fatal_lines}"
        )
        assert "host2.example.com" in fatal_lines[0], (
            "Fatal log should be from host2.example.com"
        )

    def test_selects_error_logs(self, multi_line_log_sample):
        """GIVEN multi-line logs with error entries
        WHEN select_error_logs is called
        THEN it should include error logs."""
        result = select_error_logs(multi_line_log_sample)

        error_lines = [line for line in result if "error:" in line]
        assert len(error_lines) == 1, (
            f"Expected exactly 1 error log, got {len(error_lines)}: {error_lines}"
        )
        assert "host3.example.com" in error_lines[0]

    def test_selects_failed_logs(self, multi_line_log_sample):
        """GIVEN multi-line logs with failed entries
        WHEN select_error_logs is called
        THEN it should include failed logs."""
        result = select_error_logs(multi_line_log_sample)

        failed_lines = [line for line in result if "failed:" in line]
        assert len(failed_lines) == 1, (
            f"Expected exactly 1 failed log, got {len(failed_lines)}: {failed_lines}"
        )
        assert "host4.example.com" in failed_lines[0]

    def test_excludes_ignoring_logs(self, multi_line_log_sample):
        """GIVEN multi-line logs with ...ignoring entries
        WHEN select_error_logs is called
        THEN it should exclude logs ending with ...ignoring."""
        result = select_error_logs(multi_line_log_sample)

        ignoring_lines = [line for line in result if "...ignoring" in line]
        assert len(ignoring_lines) == 0, (
            f"Logs with ...ignoring should be excluded, but found: {ignoring_lines}"
        )

    def test_excludes_success_logs(self, multi_line_log_sample):
        """GIVEN multi-line logs with ok/success entries
        WHEN select_error_logs is called
        THEN it should exclude success logs."""
        result = select_error_logs(multi_line_log_sample)

        ok_lines = [line for line in result if line.strip().startswith("ok:")]
        assert len(ok_lines) == 0, (
            f"Success logs should be excluded, but found: {ok_lines}"
        )

    def test_empty_input_returns_empty_list(self):
        """GIVEN an empty string
        WHEN select_error_logs is called
        THEN it should return an empty list."""
        result = select_error_logs("")
        assert result == [], f"Empty input should return empty list, got: {result}"

    def test_no_errors_returns_empty_list(self):
        """GIVEN logs with only success entries
        WHEN select_error_logs is called
        THEN it should return an empty list."""
        success_only = """ok: [host1.example.com] => {"changed": false}

ok: [host2.example.com] => {"msg": "Success"}

changed: [host3.example.com] => {"msg": "Updated"}"""

        result = select_error_logs(success_only)
        assert result == [], (
            f"Logs with no errors should return empty list, got: {result}"
        )

    def test_handles_single_error_line(self, fatal_log_line):
        """GIVEN a single error log line
        WHEN select_error_logs is called
        THEN it should return that line."""
        result = select_error_logs(fatal_log_line)
        assert len(result) == 1, f"Expected 1 result, got {len(result)}"
        assert result[0] == fatal_log_line

    def test_preserves_original_log_format(self):
        """GIVEN logs with special characters and formatting
        WHEN select_error_logs is called
        THEN it should preserve the exact original format."""
        special_log = 'fatal: [special-host_01.sub.domain.com]: FAILED! => {"msg": "Error with \\"quotes\\" and tabs\t"}'
        result = select_error_logs(special_log)
        assert result[0] == special_log, "Original formatting should be preserved"


# =============================================================================
# Tests for detect_error_level()
# =============================================================================


class TestDetectErrorLevel:
    """Tests for detect_error_level function - detects log severity level."""

    def test_detects_failed_level(self, failed_log_line):
        """GIVEN a log with 'failed:' prefix
        WHEN detect_error_level is called
        THEN it should return DetectedLevel.ERROR (failed maps to error)."""
        result = detect_error_level(failed_log_line)
        # 'failed' is mapped to ERROR since DetectedLevel only has error/warn/info/debug/unknown
        assert result == DetectedLevel.ERROR, (
            f"Expected DetectedLevel.ERROR for 'failed', got: {result}"
        )

    def test_detects_fatal_level(self, fatal_log_line):
        """GIVEN a log with 'fatal:' prefix
        WHEN detect_error_level is called
        THEN it should return DetectedLevel.ERROR (fatal maps to error)."""
        result = detect_error_level(fatal_log_line)
        # 'fatal' is mapped to ERROR since DetectedLevel only has error/warn/info/debug/unknown
        assert result == DetectedLevel.ERROR, (
            f"Expected DetectedLevel.ERROR for 'fatal', got: {result}"
        )

    def test_detects_error_level(self, error_log_line):
        """GIVEN a log with 'error:' prefix
        WHEN detect_error_level is called
        THEN it should return DetectedLevel.ERROR."""
        result = detect_error_level(error_log_line)
        assert result == DetectedLevel.ERROR, (
            f"Expected DetectedLevel.ERROR, got: {result}"
        )

    def test_returns_unknown_for_success_logs(self, success_log_line):
        """GIVEN a successful log without error indicators
        WHEN detect_error_level is called
        THEN it should return DetectedLevel.UNKNOWN."""
        result = detect_error_level(success_log_line)
        assert result == DetectedLevel.UNKNOWN, (
            f"Expected DetectedLevel.UNKNOWN for success log, got: {result}"
        )

    def test_returns_unknown_for_empty_string(self):
        """GIVEN an empty string
        WHEN detect_error_level is called
        THEN it should return DetectedLevel.UNKNOWN."""
        result = detect_error_level("")
        assert result == DetectedLevel.UNKNOWN, (
            f"Expected DetectedLevel.UNKNOWN for empty string, got: {result}"
        )

    def test_returns_unknown_for_random_text(self):
        """GIVEN random text without error patterns
        WHEN detect_error_level is called
        THEN it should return DetectedLevel.UNKNOWN."""
        result = detect_error_level("This is just a random message without errors")
        assert result == DetectedLevel.UNKNOWN

    def test_case_sensitivity(self):
        """GIVEN logs with uppercase error keywords
        WHEN detect_error_level is called
        THEN it should NOT match (regex is case-sensitive)."""
        uppercase_log = 'FATAL: [host]: FAILED! => {"msg": "error"}'
        result = detect_error_level(uppercase_log)
        # The regex uses lowercase, so FATAL won't match
        assert result == DetectedLevel.UNKNOWN, (
            "Uppercase FATAL should not match (case-sensitive regex)"
        )

    def test_matches_first_occurrence(self):
        """GIVEN a log with multiple error patterns
        WHEN detect_error_level is called
        THEN it should match the first occurrence."""
        multi_error = 'error: [host]: error: [host2]: FAILED! => {"msg": "nested"}'
        result = detect_error_level(multi_error)
        assert result == DetectedLevel.ERROR, "Should match first 'error:' pattern"


# =============================================================================
# Tests for get_log_message()
# =============================================================================


class TestGetLogMessage:
    """Tests for get_log_message function - extracts log message from Ansible output."""

    def test_extracts_message_from_fatal_log(self, fatal_log_line):
        """GIVEN a standard fatal log line
        WHEN get_log_message is called
        THEN it should extract the JSON message content."""
        result = extract_error_from_log(fatal_log_line)
        # The regex captures the content inside the {} after =>
        assert "Connection timeout" in result, (
            f"Expected 'Connection timeout' in result, got: {result}"
        )

    def test_extracts_message_from_failed_log_with_item(self, failed_log_line):
        """GIVEN a failed log with (item=...) format
        WHEN get_log_message is called
        THEN it should extract the message correctly."""
        result = extract_error_from_log(failed_log_line)
        assert "Package not found" in result, (
            f"Expected 'Package not found' in result, got: {result}"
        )

    def test_returns_original_when_no_match(self):
        """GIVEN a log that doesn't match the extraction pattern
        WHEN get_log_message is called
        THEN it should return the original log."""
        weird_log = "This is not a standard Ansible log format"
        result = extract_error_from_log(weird_log)
        assert result == weird_log, (
            f"Non-matching logs should return original, got: {result}"
        )

    def test_handles_empty_string(self):
        """GIVEN an empty string
        WHEN get_log_message is called
        THEN it should return the empty string."""
        result = extract_error_from_log("")
        assert result == "", f"Empty string should return empty, got: {result}"

    def test_handles_multiline_json(self):
        """GIVEN a log with multiline JSON content
        WHEN get_log_message is called
        THEN it should capture the full multiline content."""
        multiline_log = """fatal: [host1.example.com]: FAILED! => {
    "msg": "This is a multiline error message",
    "details": {
        "code": 500,
        "error": "Internal Server Error"
    }
}"""
        result = extract_error_from_log(multiline_log)
        assert "multiline error message" in result, (
            f"Should capture multiline content, got: {result}"
        )

    def test_takes_last_match_for_multiple_patterns(self):
        """GIVEN a log with multiple matching patterns (nested errors)
        WHEN get_log_message is called
        THEN it should return the last match."""
        nested_log = 'fatal: [host1]: FAILED! => {"outer": "error"} fatal: [host2]: FAILED! => {"inner": "error"}'
        result = extract_error_from_log(nested_log)
        # The code takes matches[-1] - the last match
        assert "inner" in result, f"Should take last match, got: {result}"

    def test_handles_special_characters_in_message(self):
        """GIVEN a log with special characters (quotes, backslashes)
        WHEN get_log_message is called
        THEN it should preserve special characters."""
        special_log = r'error: [host]: CHANGED! => {"msg": "Path: /var/log/\"test\".log", "rc": 0}'
        result = extract_error_from_log(special_log)
        assert "Path:" in result, (
            f"Should capture path with special chars, got: {result}"
        )

    def test_handles_ipv6_host(self):
        """GIVEN a log with IPv6 address as host
        WHEN get_log_message is called
        THEN it should extract the message correctly."""
        ipv6_log = 'fatal: [2001:db8::1]: FAILED! => {"msg": "IPv6 host error"}'
        result = extract_error_from_log(ipv6_log)
        assert "IPv6 host error" in result, f"Should handle IPv6 hosts, got: {result}"


# =============================================================================
# Tests for slice_log_message()
# =============================================================================


class TestSliceLogMessage:
    """Tests for slice_log_message function - truncates logs to 5000 chars."""

    def test_short_message_unchanged(self):
        """GIVEN a message shorter than 5000 characters
        WHEN slice_log_message is called
        THEN it should return the message unchanged (just stripped)."""
        short_msg = "This is a short message"
        result = slice_log_message(short_msg)
        assert result == short_msg.strip(), (
            f"Short message should be returned as-is, got: {result}"
        )

    def test_exact_5000_chars_unchanged(self):
        """GIVEN a message exactly 5000 characters
        WHEN slice_log_message is called
        THEN it should return all 5000 characters."""
        exact_msg = "a" * 5000
        result = slice_log_message(exact_msg)
        assert len(result) == 5000, (
            f"Exactly 5000 chars should not be truncated, got length: {len(result)}"
        )

    def test_truncates_long_message(self):
        """GIVEN a message longer than 5000 characters
        WHEN slice_log_message is called
        THEN it should truncate to exactly 5000 characters."""
        long_msg = "x" * 10000
        result = slice_log_message(long_msg)
        assert len(result) == 5000, (
            f"Long message should be truncated to 5000, got length: {len(result)}"
        )

    def test_strips_leading_whitespace_before_truncation(self):
        """GIVEN a message with leading whitespace
        WHEN slice_log_message is called
        THEN it should strip whitespace before measuring/truncating."""
        padded_msg = "   " + "a" * 5000
        result = slice_log_message(padded_msg)
        # After strip(), we have 5000 'a's, which fits exactly
        assert len(result) == 5000, (
            f"Should strip then truncate, got length: {len(result)}"
        )
        assert not result.startswith(" "), "Leading whitespace should be stripped"

    def test_strips_trailing_whitespace(self):
        """GIVEN a message with trailing whitespace
        WHEN slice_log_message is called
        THEN it should strip trailing whitespace."""
        trailing_msg = "message   \n\t"
        result = slice_log_message(trailing_msg)
        assert result == "message", (
            f"Trailing whitespace should be stripped, got: '{result}'"
        )

    def test_empty_string(self):
        """GIVEN an empty string
        WHEN slice_log_message is called
        THEN it should return an empty string."""
        result = slice_log_message("")
        assert result == "", f"Empty string should return empty, got: '{result}'"

    def test_whitespace_only(self):
        """GIVEN a string with only whitespace
        WHEN slice_log_message is called
        THEN it should return an empty string."""
        result = slice_log_message("   \n\t  ")
        assert result == "", f"Whitespace-only should return empty, got: '{result}'"


# =============================================================================
# Tests for clean_slash()
# =============================================================================


class TestCleanSlash:
    """Tests for clean_slash function - removes excess backslashes and escaped quotes."""

    def test_removes_double_backslashes(self):
        """GIVEN a string with double backslashes
        WHEN clean_slash is called
        THEN it should reduce them to single backslashes."""
        result = clean_slash("path\\\\to\\\\file")
        assert "\\\\" not in result, (
            f"Double backslashes should be reduced, got: {result}"
        )

    def test_removes_quadruple_backslashes(self):
        """GIVEN a string with quadruple backslashes
        WHEN clean_slash is called
        THEN it should reduce them to single backslashes."""
        result = clean_slash("path\\\\\\\\to\\\\\\\\file")
        assert "\\\\\\\\" not in result, (
            f"Quadruple backslashes should be reduced, got: {result}"
        )

    def test_removes_escaped_quotes(self):
        """GIVEN a string with escaped quotes
        WHEN clean_slash is called
        THEN it should convert them to regular quotes."""
        result = clean_slash('message with \\"quoted\\" text')
        assert '\\"' not in result, f"Escaped quotes should be unescaped, got: {result}"
        assert '"quoted"' in result, f"Should contain regular quotes, got: {result}"

    def test_handles_mixed_escapes(self):
        """GIVEN a string with mixed backslashes and escaped quotes
        WHEN clean_slash is called
        THEN it should clean both."""
        result = clean_slash('path\\\\to\\\\"file\\"')
        assert "\\\\" not in result
        assert '\\"' not in result

    def test_no_change_for_clean_string(self):
        """GIVEN a string without escape sequences
        WHEN clean_slash is called
        THEN it should return the string unchanged."""
        clean_str = "This is a clean string without escapes"
        result = clean_slash(clean_str)
        assert result == clean_str

    def test_empty_string(self):
        """GIVEN an empty string
        WHEN clean_slash is called
        THEN it should return an empty string."""
        result = clean_slash("")
        assert result == ""

    def test_single_backslash_preserved(self):
        """GIVEN a string with single backslashes
        WHEN clean_slash is called
        THEN single backslashes should remain."""
        result = clean_slash("path\\to\\file")
        assert result == "path\\to\\file"


# =============================================================================
# Tests for pre_proccess_log_without_extraction()
# =============================================================================


class TestPreProccessLogWithoutExtraction:
    """Tests for pre_proccess_log_without_extraction - cleans and slices without extraction."""

    def test_cleans_and_slices(self):
        """GIVEN a log with escape sequences
        WHEN pre_proccess_log_without_extraction is called
        THEN it should clean slashes and truncate."""
        log = 'Some log with \\"escaped\\" quotes'
        result = pre_proccess_log_without_extraction(log)
        assert '\\"' not in result, "Escaped quotes should be cleaned"
        assert len(result) <= 5000

    def test_long_message_truncated(self):
        """GIVEN a very long log message
        WHEN pre_proccess_log_without_extraction is called
        THEN it should truncate to 5000 characters."""
        long_log = "x" * 10000
        result = pre_proccess_log_without_extraction(long_log)
        assert len(result) == 5000

    def test_strips_whitespace(self):
        """GIVEN a log with leading/trailing whitespace
        WHEN pre_proccess_log_without_extraction is called
        THEN it should strip whitespace."""
        log = "   message with whitespace   "
        result = pre_proccess_log_without_extraction(log)
        assert result == "message with whitespace"

    def test_empty_string(self):
        """GIVEN an empty string
        WHEN pre_proccess_log_without_extraction is called
        THEN it should return an empty string."""
        result = pre_proccess_log_without_extraction("")
        assert result == ""

    def test_preserves_structure(self):
        """GIVEN a full Ansible log line
        WHEN pre_proccess_log_without_extraction is called
        THEN it should preserve the log structure (not extract)."""
        log = 'fatal: [host]: FAILED! => {"msg": "error"}'
        result = pre_proccess_log_without_extraction(log)
        assert "fatal:" in result, "Should preserve log structure"
        assert "[host]" in result, "Should preserve host"


# =============================================================================
# Tests for filter_ingoring()
# =============================================================================


class TestFilterIgnoring:
    """Tests for filter_ingoring function - checks for ...ignoring suffix."""

    def test_returns_true_for_ignoring_log(self, ignoring_log_line):
        """GIVEN a log containing '...ignoring'
        WHEN filter_ingoring is called
        THEN it should return True."""
        result = filter_ingoring(ignoring_log_line)
        assert result is True, (
            f"Logs with ...ignoring should return True, got: {result}"
        )

    def test_returns_false_for_normal_log(self, fatal_log_line):
        """GIVEN a log without '...ignoring'
        WHEN filter_ingoring is called
        THEN it should return False."""
        result = filter_ingoring(fatal_log_line)
        assert result is False, f"Normal logs should return False, got: {result}"

    def test_returns_false_for_empty_string(self):
        """GIVEN an empty string
        WHEN filter_ingoring is called
        THEN it should return False."""
        result = filter_ingoring("")
        assert result is False, f"Empty string should return False, got: {result}"

    def test_ignoring_anywhere_in_string(self):
        """GIVEN a log with '...ignoring' in the middle
        WHEN filter_ingoring is called
        THEN it should still return True."""
        middle_ignoring = "start of log ...ignoring more content after"
        result = filter_ingoring(middle_ignoring)
        assert result is True, "...ignoring anywhere in string should return True"

    def test_partial_ignoring_not_matched(self):
        """GIVEN a log with partial match like 'ignoring' without dots
        WHEN filter_ingoring is called
        THEN it should return False (exact '...ignoring' required)."""
        partial = "This task is ignoring the result"
        result = filter_ingoring(partial)
        # The function checks for "...ignoring" exactly
        assert result is False, "Partial 'ignoring' without dots should return False"

    def test_case_sensitivity(self):
        """GIVEN a log with uppercase '...IGNORING'
        WHEN filter_ingoring is called
        THEN it should return False (case-sensitive)."""
        uppercase = "fatal: [host]: FAILED! => {} ...IGNORING"
        result = filter_ingoring(uppercase)
        assert result is False, (
            "...IGNORING uppercase should not match (case-sensitive)"
        )


# =============================================================================
# Tests for pre_proccess_log()
# =============================================================================


class TestPreProccessLog:
    """Tests for pre_proccess_log function - combines extraction, cleaning, and slicing."""

    def test_extracts_cleans_and_truncates(self, fatal_log_line):
        """GIVEN a standard fatal log line
        WHEN pre_proccess_log is called
        THEN it should extract message, clean slashes, and apply truncation."""
        result = pre_proccess_log(fatal_log_line)
        # Should have extracted the JSON content
        assert "Connection timeout" in result
        # Should not exceed 5000 chars
        assert len(result) <= 5000

    def test_cleans_escaped_quotes_in_extracted_message(self):
        """GIVEN a log with escaped quotes in the message
        WHEN pre_proccess_log is called
        THEN it should extract and clean the escaped quotes."""
        log = r'fatal: [host]: FAILED! => {"msg": "Error with \"quoted\" text"}'
        result = pre_proccess_log(log)
        assert '\\"' not in result, "Escaped quotes should be cleaned"

    def test_long_message_truncated(self):
        """GIVEN a log with very long message content
        WHEN pre_proccess_log is called
        THEN it should truncate to 5000 characters."""
        long_content = "x" * 10000
        long_log = f'fatal: [host]: FAILED! => {{"msg": "{long_content}"}}'
        result = pre_proccess_log(long_log)
        assert len(result) <= 5000, (
            f"Result should be truncated to 5000, got length: {len(result)}"
        )

    def test_non_matching_log_returned_truncated(self):
        """GIVEN a log that doesn't match the extraction pattern
        WHEN pre_proccess_log is called
        THEN it should return original (truncated if needed)."""
        non_matching = "Just a random log message " * 500
        result = pre_proccess_log(non_matching)
        # get_log_message returns original, then slice_log_message truncates
        assert len(result) <= 5000


# =============================================================================
# Tests for proccess_log_inference()
# =============================================================================


class TestProccessLogInference:
    """Tests for proccess_log_inference function - conditional processing."""

    def test_task_prefix_uses_pre_proccess(self):
        """GIVEN a log starting with 'TASK'
        WHEN proccess_log_inference is called
        THEN it should use pre_proccess_log (extract + slice)."""
        task_log = (
            'TASK [Install package] fatal: [host]: FAILED! => {"msg": "Install failed"}'
        )
        result = proccess_log_inference(task_log)
        # Should extract the message content
        assert "Install failed" in result

    def test_non_task_uses_clean_and_slice_only(self):
        """GIVEN a log NOT starting with 'TASK'
        WHEN proccess_log_inference is called
        THEN it should clean slashes and slice (not extract)."""
        non_task = 'fatal: [host]: FAILED! => {"msg": "Direct error"}'
        result = proccess_log_inference(non_task)
        # pre_proccess_log_without_extraction is applied, so the whole log structure
        # is preserved (cleaned and potentially truncated)
        assert "fatal:" in result

    def test_empty_string(self):
        """GIVEN an empty string
        WHEN proccess_log_inference is called
        THEN it should return empty string."""
        result = proccess_log_inference("")
        assert result == ""

    def test_task_case_sensitivity(self):
        """GIVEN a log starting with lowercase 'task'
        WHEN proccess_log_inference is called
        THEN it should NOT use pre_proccess_log (uses slice only)."""
        lowercase_task = 'task [setup] fatal: [host]: FAILED! => {"msg": "error"}'
        result = proccess_log_inference(lowercase_task)
        # Since it uses startswith("TASK"), lowercase won't match
        assert "task [setup]" in result, (
            "Lowercase 'task' should not trigger pre_proccess_log"
        )

    def test_long_non_task_log_truncated(self):
        """GIVEN a long log not starting with TASK
        WHEN proccess_log_inference is called
        THEN it should truncate to 5000 characters."""
        long_log = "ERROR: " + "x" * 10000
        result = proccess_log_inference(long_log)
        assert len(result) <= 5000

    def test_non_task_cleans_escaped_quotes(self):
        """GIVEN a non-TASK log with escaped quotes
        WHEN proccess_log_inference is called
        THEN it should clean the escaped quotes."""
        log = r'fatal: [host]: FAILED! => {"msg": \"error\"}'
        result = proccess_log_inference(log)
        assert '\\"' not in result, "Escaped quotes should be cleaned"

    def test_task_cleans_escaped_quotes(self):
        """GIVEN a TASK log with escaped quotes in extracted content
        WHEN proccess_log_inference is called
        THEN it should extract and clean escaped quotes."""
        log = r'TASK [test] fatal: [host]: FAILED! => {"msg": "Error with \"quotes\""}'
        result = proccess_log_inference(log)
        assert '\\"' not in result, "Escaped quotes should be cleaned"


# =============================================================================
# Integration / Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Edge cases and integration tests for transformation functions."""

    def test_complete_workflow_with_multi_line_logs(self, multi_line_log_sample):
        """GIVEN a realistic multi-line Ansible log output
        WHEN the complete transformation workflow is applied
        THEN all functions should work correctly together."""
        # Step 1: Select error logs
        error_logs = select_error_logs(multi_line_log_sample)
        assert len(error_logs) >= 3, "Should find at least 3 error logs"

        for log in error_logs:
            # Step 2: Detect level - all error patterns (fatal/failed/error) map to ERROR
            level = detect_error_level(log)
            assert level == DetectedLevel.ERROR, (
                f"Error log should have DetectedLevel.ERROR: {log}"
            )

            # Step 3: Check not ignoring
            assert not filter_ingoring(log), (
                f"...ignoring logs should have been filtered: {log}"
            )

            # Step 4: Process the log
            processed = pre_proccess_log(log)
            assert len(processed) <= 5000
            assert len(processed) > 0

    def test_unicode_and_special_characters_throughout(self):
        """GIVEN logs with unicode and special characters
        WHEN all transformation functions are applied
        THEN they should handle special chars correctly."""
        unicode_log = (
            'fatal: [サーバー.example.com]: FAILED! => {"msg": "Error: 文字化け 🔥"}'
        )

        # Should not raise exceptions
        level = detect_error_level(unicode_log)
        assert level is not None

        message = extract_error_from_log(unicode_log)
        assert "文字化け" in message or unicode_log == message

        sliced = slice_log_message(message)
        assert len(sliced) <= 5000

    def test_extremely_long_single_line_log(self):
        """GIVEN an extremely long single-line log
        WHEN transformation functions are applied
        THEN they should handle it without errors or memory issues."""
        # 100KB log message
        huge_msg = "x" * 100_000
        huge_log = f'fatal: [host]: FAILED! => {{"msg": "{huge_msg}"}}'

        # Should not raise
        result = pre_proccess_log(huge_log)
        assert len(result) == 5000, "Should truncate to exactly 5000"

    def test_empty_json_in_log(self):
        """GIVEN a log with empty JSON content
        WHEN get_log_message is called
        THEN it should handle empty content gracefully."""
        empty_json = "fatal: [host]: FAILED! => {}"
        result = extract_error_from_log(empty_json)
        # The regex captures content inside {}, which is empty
        # Depending on regex behavior, might return empty or original
        assert result is not None

    def test_malformed_json_in_log(self):
        """GIVEN a log with malformed JSON
        WHEN get_log_message is called
        THEN it should not crash and return something sensible."""
        malformed = 'fatal: [host]: FAILED! => {"msg": broken json here'
        result = extract_error_from_log(malformed)
        # Should not raise, should return something
        assert result is not None
        assert len(result) > 0


# =============================================================================
# Boundary Condition Tests
# =============================================================================


class TestBoundaryConditions:
    """Tests for boundary conditions and limits."""

    def test_slice_at_exactly_5000_boundary(self):
        """GIVEN messages at exactly 4999, 5000, and 5001 characters
        WHEN slice_log_message is called
        THEN boundary behavior should be correct."""
        msg_4999 = "a" * 4999
        msg_5000 = "a" * 5000
        msg_5001 = "a" * 5001

        assert len(slice_log_message(msg_4999)) == 4999
        assert len(slice_log_message(msg_5000)) == 5000
        assert len(slice_log_message(msg_5001)) == 5000

    def test_regex_with_various_host_formats(self):
        """GIVEN logs with various host name formats
        WHEN get_log_message is called
        THEN all valid host formats should be parsed."""
        hosts = [
            "simple",
            "host.domain.com",
            "host-with-dashes",
            "host_with_underscores",
            "192.168.1.1",
            "host123",
            "UPPERCASE.HOST",
        ]

        for host in hosts:
            log = f'fatal: [{host}]: FAILED! => {{"msg": "test error for {host}"}}'
            result = extract_error_from_log(log)
            assert f"test error for {host}" in result, (
                f"Failed to extract message for host: {host}"
            )

    def test_newline_variations(self):
        """GIVEN logs separated by different newline types
        WHEN select_error_logs is called
        THEN it should handle all newline variations."""
        # Unix newlines
        unix = "ok: [h1]\n\nfatal: [h2]: FAILED! => {}"
        assert len(select_error_logs(unix)) >= 1

        # The function splits on \n\n, so other variations might behave differently
        # This documents the expected behavior
