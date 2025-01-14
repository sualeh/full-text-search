package us.fatehi.search;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.StoredValue;
import org.apache.lucene.document.StoredValue.Type;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexableField;
import org.apache.lucene.index.StoredFields;
import org.apache.lucene.queryparser.classic.ParseException;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.BooleanClause.Occur;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.FieldDoc;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.MatchAllDocsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import static java.util.Objects.requireNonNull;
import dev.langchain4j.data.document.Metadata;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.rag.content.Content;
import dev.langchain4j.rag.content.ContentMetadata;
import dev.langchain4j.rag.content.retriever.ContentRetriever;

/** Full-text content retrieval using Apache Lucene for LangChain4J RAG. */
public final class LuceneContentRetriever implements ContentRetriever {

  /** Builder for `LuceneContentRetriever`. */
  public static class LuceneContentRetrieverBuilder {

    private Directory directory;
    private boolean onlyMatches;
    private int topNMatches;
    private int maxTokens;

    private LuceneContentRetrieverBuilder() {
      // Set defaults
      onlyMatches = true;
      topNMatches = 10;
      maxTokens = Integer.MAX_VALUE;
    }

    /**
     * Build an instance of `LuceneContentRetriever` using internal builder field values.
     *
     * @return New instance of `LuceneContentRetriever`
     */
    public LuceneContentRetriever build() {
      if (directory == null) {
        directory = DirectoryFactory.tempDirectory();
      }
      return new LuceneContentRetriever(directory, onlyMatches, topNMatches, maxTokens);
    }

    /**
     * Sets the Lucene directory. If null, a temporary file-based directory is used.
     *
     * @param directory Lucene directory
     * @return Builder
     */
    public LuceneContentRetrieverBuilder directory(final Directory directory) {
      // Can be null
      this.directory = directory;
      return this;
    }

    /**
     * Provides documents until the top N, even if there is no good match.
     *
     * @return Builder
     */
    public LuceneContentRetrieverBuilder matchUntilTopN() {
      onlyMatches = false;
      return this;
    }

    /**
     * Returns documents until the maximum token limit is reached.
     *
     * @param maxTokens Maximum number of tokens
     * @return Builder
     */
    public LuceneContentRetrieverBuilder maxTokens(final int maxTokens) {
      if (maxTokens >= 0) {
        this.maxTokens = maxTokens;
      }
      return this;
    }

    /**
     * Provides only documents matched to the query using full text search.
     *
     * @return Builder
     */
    public LuceneContentRetrieverBuilder onlyMatches() {
      onlyMatches = true;
      return this;
    }

    /**
     * Returns only a certain number of documents.
     *
     * @param topNMatches Number of documents to return
     * @return Builder
     */
    public LuceneContentRetrieverBuilder topNMatches(final int topNMatches) {
      if (topNMatches >= 0) {
        this.topNMatches = topNMatches;
      }
      return this;
    }
  }

  private static final Logger LOGGER =
      Logger.getLogger(LuceneContentRetriever.class.getCanonicalName());

  /**
   * Instantiate a builder for `LuceneContentRetriever`.
   *
   * @return Builder for `LuceneContentRetriever`
   */
  public static LuceneContentRetrieverBuilder builder() {
    return new LuceneContentRetrieverBuilder();
  }

  private final Directory directory;
  private final boolean onlyMatches;
  private final int topNMatches;
  private final int maxTokens;

  private LuceneContentRetriever(
      final Directory directory,
      final boolean onlyMatches,
      final int topNMatches,
      final int maxTokens) {
    this.directory = requireNonNull(directory, "No directory provided");
    this.onlyMatches = onlyMatches;
    this.topNMatches = Math.max(0, topNMatches);
    this.maxTokens = Math.max(0, maxTokens);
  }

  /** {@inheritDoc} */
  @Override
  public List<Content> retrieve(final dev.langchain4j.rag.query.Query query) {
    if (query == null) {
      return Collections.emptyList();
    }

    int docCount = 0;
    int tokenCount = 0;
    try (final DirectoryReader reader = DirectoryReader.open(directory)) {

      final Query luceneQuery = buildQuery(query.text());

      final IndexSearcher searcher = new IndexSearcher(reader);
      final TopDocs topDocs = searcher.search(luceneQuery, topNMatches, Sort.RELEVANCE);
      final List<Content> hits = new ArrayList<>();
      final StoredFields storedFields = reader.storedFields();
      for (final ScoreDoc scoreDoc : topDocs.scoreDocs) {
        // Check if number of documents is exceeded
        docCount = docCount + 1;
        if (docCount > topNMatches) {
          break;
        }

        // Retrieve document contents
        final Document document = storedFields.document(scoreDoc.doc);

        // Check if maximum token count is exceeded
        final IndexableField tokenCountField = document.getField(LuceneIndexer.TOKEN_COUNT);
        final int docTokens = tokenCountField.numericValue().intValue();
        if (tokenCount + docTokens > maxTokens) {
          continue;
          // There may be smaller documents to come after this that we can accommodate
        }
        tokenCount = tokenCount + docTokens;

        // Add all other document fields to metadata
        final Metadata metadata = createTextSegmentMetadata(document);

        // Finally create the text segment
        final TextSegment textSegment =
            TextSegment.from(document.get(LuceneIndexer.CONTENT), metadata);

        hits.add(Content.from(textSegment, withScore(scoreDoc)));
      }
      return hits;
    } catch (final Exception e) {
      LOGGER.log(Level.INFO, String.format("Could not query <%s>", query), e);
      return Collections.emptyList();
    }
  }

  private Query buildQuery(final String query) throws ParseException {
    final QueryParser parser = new QueryParser(LuceneIndexer.CONTENT, new StandardAnalyzer());
    final Query fullTextQuery = parser.parse(query);
    if (onlyMatches) {
      return fullTextQuery;
    }

    final BooleanQuery combinedQuery =
        new BooleanQuery.Builder()
            .add(fullTextQuery, Occur.SHOULD)
            .add(new MatchAllDocsQuery(), Occur.SHOULD)
            .build();
    return combinedQuery;
  }

  private Metadata createTextSegmentMetadata(final Document document) {
    final Metadata metadata = new Metadata();
    for (final IndexableField field : document) {
      final String fieldName = field.name();
      if (LuceneIndexer.CONTENT.equals(fieldName)) {
        continue;
      }

      final StoredValue storedValue = field.storedValue();
      final Type type = storedValue.getType();
      switch (type) {
        case INTEGER:
          metadata.put(fieldName, storedValue.getIntValue());
          break;
        case LONG:
          metadata.put(fieldName, storedValue.getLongValue());
          break;
        case FLOAT:
          metadata.put(fieldName, storedValue.getFloatValue());
          break;
        case DOUBLE:
          metadata.put(fieldName, storedValue.getDoubleValue());
          break;
        case STRING:
          metadata.put(fieldName, storedValue.getStringValue());
          break;
        default:
          // No-op
      }
    }
    return metadata;
  }

  private Map<ContentMetadata, Object> withScore(final ScoreDoc scoreDoc) {
    final Map<ContentMetadata, Object> contentMetadata = new HashMap<>();
    try {
      contentMetadata.put(ContentMetadata.SCORE, (float) ((FieldDoc) scoreDoc).fields[0] - 1f);
    } catch (final Exception e) {
      // Ignore = No score will be added to content metadata
    }
    return contentMetadata;
  }
}
