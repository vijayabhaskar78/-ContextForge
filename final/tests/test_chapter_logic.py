import sys
import unittest
from pathlib import Path


from contextforge import app


class TestChapterIntentRouting(unittest.TestCase):
    # Exactly 100 intent-routing cases.
    CASES = [
        # Count intent (20)
        ("How many chapters are there", "count"),
        ("how many chpaters are there", "count"),
        ("chapter count please", "count"),
        ("number of chapters in this book", "count"),
        ("just tell me the count of chapters", "count"),
        ("i need chapter count", "count"),
        ("count chapters", "count"),
        ("how many number of chpaters are there i need the count", "count"),
        ("total chapters?", "count"),
        ("chapter total", "count"),
        ("how many chaptrs", "count"),
        ("how many chaptors in this pdf", "count"),
        ("how many chapter are there", "count"),
        ("count of chpaters", "count"),
        ("number of chpaters", "count"),
        ("count chapers", "count"),
        ("tell chapters count", "count"),
        ("need chapters count now", "count"),
        ("chapters count?", "count"),
        ("total number of chapter in pdf", "count"),

        # List intent (20)
        ("list all chapters", "list"),
        ("show chapters", "list"),
        ("give me chapters", "list"),
        ("what chapters are there", "list"),
        ("which chapters are in this pdf", "list"),
        ("show table of contents", "list"),
        ("toc please", "list"),
        ("contents of this book", "list"),
        ("list chapter names", "list"),
        ("display all chapter titles", "list"),
        ("list chapters", "list"),
        ("show me all chapters", "list"),
        ("give all chapter names", "list"),
        ("chapter list please", "list"),
        ("all chapters in this book", "list"),
        ("toc", "list"),
        ("table of contents please", "list"),
        ("contents page chapters", "list"),
        ("which chapters exist", "list"),
        ("chapter titles list", "list"),

        # Lookup intent (20)
        ("what is the name of 20th chapter", "number_lookup"),
        ("name of chapter 20", "number_lookup"),
        ("what is 20th chapter called", "number_lookup"),
        ("which chapter is deployment", "lookup"),
        ("what chapter is deployment", "lookup"),
        ("chapter number for deployment", "lookup"),
        ("which chapter covers rag", "lookup"),
        ("title of chapter 31", "number_lookup"),
        ("which one is chapter deployment", "lookup"),
        ("find chapter for multimodal", "lookup"),
        ("name of 1st chapter", "number_lookup"),
        ("name of 2nd chapter", "number_lookup"),
        ("name of 3rd chapter", "number_lookup"),
        ("name of 4th chapter", "number_lookup"),
        ("what is 10th chapter", "number_lookup"),
        ("what is 34th chapter", "number_lookup"),
        ("which chapter is evals", "lookup"),
        ("which chapter is multimodal", "lookup"),
        ("chapter number for deployment topic", "lookup"),
        ("which chapter has workflows as tools", "lookup"),

        # Explain intent (20)
        ("explain chapter 20", "explain"),
        ("explain me chapter 20", "explain"),
        ("summarize chapter 31", "explain"),
        ("summary of chapter 31", "explain"),
        ("what is in chapter 31", "explain"),
        ("what is in chapter 31 deployment", "explain"),
        ("tell me chapter 20", "explain"),
        ("describe chapter 20", "explain"),
        ("about chapter 20", "explain"),
        ("can you explain me the chaper deployment", "explain"),
        ("explain chapter 1", "explain"),
        ("explain chapter 2", "explain"),
        ("explain chapter 10", "explain"),
        ("explain chapter 34", "explain"),
        ("summarize chaper 20", "explain"),
        ("summary chapter 22", "explain"),
        ("describe chapter 18", "explain"),
        ("what does chapter 21 say", "explain"),
        ("tell me chapter 5 details", "explain"),
        ("about chpatr 27", "explain"),

        # Non-chapter/general intent (20)
        ("who is the author", "none"),
        ("summarize this pdf", "none"),
        ("what is rag", "none"),
        ("what are tools", "none"),
        ("tell me about deployment without chapter mention", "none"),
        ("show me references", "none"),
        ("give me key points", "none"),
        ("what is this book about", "none"),
        ("list sections", "none"),
        ("what is model context protocol", "none"),
        ("who wrote this", "none"),
        ("what year was this published", "none"),
        ("give me quotes", "none"),
        ("find page where rag is defined", "none"),
        ("what is mcp", "none"),
        ("show key takeaways", "none"),
        ("is this book good", "none"),
        ("hello", "none"),
        ("thanks", "none"),
        ("bye", "none"),
    ]

    def test_case_count(self):
        self.assertEqual(len(self.CASES), 100, "Expected exactly 100 routing cases.")

    def test_chapter_intent_routing(self):
        failures = []
        for idx, (query, expected) in enumerate(self.CASES, start=1):
            got = app.classify_chapter_intent(query)
            if got != expected:
                failures.append(f"{idx}. query={query!r} expected={expected!r} got={got!r}")
        if failures:
            self.fail("Intent routing mismatches:\n" + "\n".join(failures))

    def test_what_is_chapter_number_routes_to_number_lookup(self):
        self.assertEqual(app.classify_chapter_intent("What is chapter 19?"), "number_lookup")


if __name__ == "__main__":
    unittest.main()
