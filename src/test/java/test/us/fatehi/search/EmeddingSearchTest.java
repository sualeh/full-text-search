package test.us.fatehi.search;

import static org.assertj.core.api.Assertions.assertThat;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import org.apache.lucene.store.Directory;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.rag.content.Content;
import dev.langchain4j.rag.content.ContentMetadata;
import dev.langchain4j.rag.query.Query;
import test.utility.StringEmbedding;
import us.fatehi.search.DirectoryFactory;
import us.fatehi.search.LuceneContentRetriever;
import us.fatehi.search.LuceneEmbeddingStore;

public class EmeddingSearchTest {

  private static final StringEmbedding[] hits = {
    StringEmbedding.fromResource("hitDoc1.txt"),
    StringEmbedding.fromResource("hitDoc2.txt"),
    StringEmbedding.fromResource("hitDoc3.txt"),
  };
  private static final StringEmbedding[] misses = {
    StringEmbedding.fromResource("missDoc1.txt"),
  };

  private Directory directory;
  private LuceneEmbeddingStore indexer;
  private LuceneContentRetriever contentRetriever;

  private static final Query query =
      Query.from(StringEmbedding.fromResource("query.txt").text().text());

  @Test
  public void queryHybrid() {

    contentRetriever = LuceneContentRetriever.builder().directory(directory).build();

    final List<String> expectedTextSegments = new ArrayList<>();
    for (final StringEmbedding stringEmbedding : hits) {
      indexer.add(stringEmbedding.embedding(), stringEmbedding.text());
      expectedTextSegments.add(stringEmbedding.text().text());
    }
    for (final StringEmbedding stringEmbedding : misses) {
      indexer.add(stringEmbedding.embedding(), stringEmbedding.text());
    }

    final List<Content> results = contentRetriever.retrieve(query);
    final List<String> actualTextSegments =
        results.stream().map(content -> content.textSegment().text()).collect(Collectors.toList());

    assertThat(actualTextSegments).hasSize(0);
  }

  @Test
  public void queryFullText() {

    contentRetriever =
        LuceneContentRetriever.builder().matchUntilTopN().directory(directory).build();

    for (final StringEmbedding stringEmbedding : hits) {
      indexer.add(null, stringEmbedding.text());
    }
    for (final StringEmbedding stringEmbedding : misses) {
      indexer.add(null, stringEmbedding.text());
    }

    System.out.printf("{\"query\"=\"%s\"}%n", query.text());
    final List<Content> results = contentRetriever.retrieve(query);
    final List<String> actualTextSegments =
        results.stream()
            .map(
                content -> {
                  final TextSegment textSegment = content.textSegment();
                  System.out.printf(
                      "{\"score\"=\"%s\", \"text\"=\"%s\"}%n",
                      content.metadata().get(ContentMetadata.SCORE), textSegment.text());
                  return textSegment.text();
                })
            .collect(Collectors.toList());

    assertThat(actualTextSegments).hasSize(4);
  }

  @BeforeEach
  public void setUp() {
    directory = DirectoryFactory.tempDirectory();
    indexer = LuceneEmbeddingStore.builder().directory(directory).build();
  }

  @AfterEach
  public void tearDown() throws Exception {
    directory.close();
  }
}
