package test.utility;

import java.util.ArrayList;
import java.util.List;
import java.util.Map.Entry;
import java.util.SortedMap;
import java.util.TreeMap;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.StoredFields;
import org.apache.lucene.queryparser.classic.ParseException;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.BooleanClause.Occur;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.TopFieldDocs;
import org.apache.lucene.store.Directory;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import dev.langchain4j.store.embedding.CosineSimilarity;
import us.fatehi.search.DirectoryFactory;
import us.fatehi.search.LuceneEmbeddingStore;

public class EmbeddingSearchResearch {

  private static final StringEmbedding[] hits = {
    StringEmbedding.fromResource("hitDoc1.txt"),
    StringEmbedding.fromResource("hitDoc2.txt"),
    StringEmbedding.fromResource("hitDoc3.txt"),
  };
  private static final StringEmbedding[] misses = {
    StringEmbedding.fromResource("missDoc1.txt"),
  };
  private static final StringEmbedding[] queries = {
    StringEmbedding.fromResource("query1.txt"),
    StringEmbedding.fromResource("query2.txt"),
    StringEmbedding.fromResource("query3.txt"),
  };

  private Directory directory;
  private LuceneEmbeddingStore indexer;

  @Test
  public void cosine() throws Exception {
    System.out.println("\n>> Cosine similarities");

    for (final StringEmbedding query : queries) {
      System.out.printf("%n{\"query\"=\"%s\"}%n", query.text());
      final SortedMap<Double, String> map = new TreeMap<>((a, b) -> b.compareTo(a));
      for (final StringEmbedding stringEmbedding : hits) {
        final double similarity =
            CosineSimilarity.between(query.embedding(), stringEmbedding.embedding());
        map.put(similarity, stringEmbedding.text().text());
      }
      for (final StringEmbedding stringEmbedding : misses) {
        final double similarity =
            CosineSimilarity.between(query.embedding(), stringEmbedding.embedding());
        map.put(similarity, stringEmbedding.text().text());
      }
      for (final Entry<Double, String> entry : map.entrySet()) {
        System.out.printf("%.4f %s%n", entry.getKey(), entry.getValue());
      }
    }
  }

  @Test
  public void queryEmbedding() throws Exception {
    System.out.println("\n>> Embedding vector query");

    for (final StringEmbedding stringEmbedding : hits) {
      indexer.add(stringEmbedding.embedding());
    }
    for (final StringEmbedding stringEmbedding : misses) {
      indexer.add(stringEmbedding.embedding());
    }

    for (final StringEmbedding query : queries) {
      System.out.printf("%n{\"query\"=\"%s\"}%n", query.text());
      final org.apache.lucene.search.Query luceneQuery = buildQuery(query);
      retrieve(luceneQuery);
    }
  }

  @Test
  public void queryFullText() throws Exception {
    System.out.println("\n>> Full text query");

    for (final StringEmbedding stringEmbedding : hits) {
      indexer.add(stringEmbedding.text());
    }
    for (final StringEmbedding stringEmbedding : misses) {
      indexer.add(stringEmbedding.text());
    }

    for (final StringEmbedding query : queries) {
      System.out.printf("%n{\"query\"=\"%s\"}%n", query.text());
      final org.apache.lucene.search.Query luceneQuery = buildQuery(query);
      retrieve(luceneQuery);

      // assertThat(actualTextSegments).hasSize(4);
    }
  }

  @Test
  public void queryHybrid() throws Exception {
    System.out.println("\n>> Hybrid query");

    for (final StringEmbedding stringEmbedding : hits) {
      indexer.add(stringEmbedding.embedding(), stringEmbedding.text());
    }
    for (final StringEmbedding stringEmbedding : misses) {
      indexer.add(stringEmbedding.embedding(), stringEmbedding.text());
    }

    for (final StringEmbedding query : queries) {
      System.out.printf("%n{\"query\"=\"%s\"}%n", query.text());
      final org.apache.lucene.search.Query luceneQuery = buildQuery(query);
      retrieve(luceneQuery);
    }
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

  private org.apache.lucene.search.Query buildQuery(final StringEmbedding query)
      throws ParseException {
    org.apache.lucene.search.Query fullTextQuery;
    final QueryParser parser = new QueryParser("content", new StandardAnalyzer());
    fullTextQuery = parser.parse(query.text().text());

    final org.apache.lucene.search.Query vectorQuery =
        // new FloatVectorSimilarityQuery("embedding", query.embedding().vector(), 0.5f);
        new KnnFloatVectorQuery("embedding", query.embedding().vector(), 5);

    final BooleanQuery combinedQuery =
        new BooleanQuery.Builder()
            .add(vectorQuery, Occur.SHOULD)
            .add(fullTextQuery, Occur.SHOULD)
            .build();

    return combinedQuery;
  }

  private List<String> retrieve(final org.apache.lucene.search.Query luceneQuery) throws Exception {
    try (final DirectoryReader reader = DirectoryReader.open(directory)) {
      final IndexSearcher searcher = new IndexSearcher(reader);
      final TopFieldDocs topDocs = searcher.search(luceneQuery, 10, Sort.RELEVANCE, true);
      final List<String> hits = new ArrayList<>();
      final StoredFields storedFields = reader.storedFields();
      for (final ScoreDoc scoreDoc : topDocs.scoreDocs) {
        // Retrieve document contents
        final Document document = storedFields.document(scoreDoc.doc);
        final String content = document.get("content");
        if (content == null || content.isBlank()) {
          continue;
        }

        System.out.printf("{\"score\"=\"%f\", \"text\"=\"%s\"}%n", scoreDoc.score, content);
        hits.add(content);
      }
      return hits;
    }
  }
}
