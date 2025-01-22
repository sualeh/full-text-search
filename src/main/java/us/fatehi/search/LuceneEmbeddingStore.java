package us.fatehi.search;

import static dev.langchain4j.internal.Utils.isNullOrEmpty;
import static dev.langchain4j.internal.Utils.randomUUID;
import static dev.langchain4j.internal.ValidationUtils.ensureNotNull;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.DoubleField;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.Field.Store;
import org.apache.lucene.document.FloatField;
import org.apache.lucene.document.IntField;
import org.apache.lucene.document.LongField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.store.Directory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import com.knuddels.jtokkit.Encodings;
import com.knuddels.jtokkit.api.Encoding;
import com.knuddels.jtokkit.api.EncodingRegistry;
import com.knuddels.jtokkit.api.EncodingType;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingSearchResult;
import dev.langchain4j.store.embedding.EmbeddingStore;

/** Lucene indexer for LangChain4J content (in the form of `TextSegment`). */
public final class LuceneEmbeddingStore implements EmbeddingStore<TextSegment> {

  static final String ID_FIELD_NAME = "id";
  static final String CONTENT_FIELD_NAME = "content";
  static final String TOKEN_COUNT_FIELD_NAME = "token-count";

  private static final Logger log = LoggerFactory.getLogger(LuceneEmbeddingStore.class);

  private final Directory directory;
  private final Encoding encoding;

  /**
   * Instantiate a new indexer to add content to an index based on a Lucene directory.
   *
   * @param directory Lucene directory
   */
  public LuceneEmbeddingStore(final Directory directory) {
    this.directory = ensureNotNull(directory, "directory");
    final EncodingRegistry registry = Encodings.newDefaultEncodingRegistry();
    encoding = registry.getEncoding(EncodingType.CL100K_BASE);
  }

  /** {@inheritDoc} */
  @Override
  public String add(final Embedding embedding) {
    final String id = randomUUID();
    add(id, embedding, null);
    return id;
  }

  /** {@inheritDoc} */
  @Override
  public String add(final Embedding embedding, final TextSegment textSegment) {
    final String id = randomUUID();
    add(id, embedding, textSegment);
    return id;
  }

  /** {@inheritDoc} */
  @Override
  public void add(final String id, final Embedding embedding) {
    add(id, embedding, null);
  }

  /**
   * Add content to a Lucene index, including segment including metadata and token count. <br>
   * IMPORTANT: Token counts are approximate, and do not include metadata.
   *
   * @param content Text segment including metadata
   */
  public void add(final String id, final Embedding embedding, final TextSegment content) {

    final IndexWriterConfig config = new IndexWriterConfig(new StandardAnalyzer());
    try (final IndexWriter writer = new IndexWriter(directory, config); ) {
      final String text = content.text();
      final int tokens = encoding.countTokens(text);

      final Document doc = new Document();
      if (isBlank(id)) {
        doc.add(new TextField(ID_FIELD_NAME, randomUUID(), Store.YES));
      } else {
        doc.add(new TextField(ID_FIELD_NAME, id, Store.YES));
      }
      if (!isBlank(text)) {
        doc.add(new TextField(CONTENT_FIELD_NAME, text, Store.YES));
      }
      doc.add(new IntField(TOKEN_COUNT_FIELD_NAME, tokens, Store.YES));

      final Map<String, Object> metadataMap = content.metadata().toMap();
      if (metadataMap != null) {
        for (final Entry<String, Object> entry : metadataMap.entrySet()) {
          doc.add(toField(entry));
        }
      }

      writer.addDocument(doc);
    } catch (final IOException e) {
      log.error(String.format("Could not write content%n%s", content), e);
    }
  }

  public String add(final TextSegment textSegment) {
    final String id = randomUUID();
    add(id, null, textSegment);
    return id;
  }

  /** {@inheritDoc} */
  @Override
  public List<String> addAll(final List<Embedding> embeddings) {
    if (embeddings == null) {
      return Collections.emptyList();
    }
    final List<String> ids = generateIds(embeddings.size());
    addAll(ids, embeddings, null);
    return ids;
  }

  /** {@inheritDoc} */
  @Override
  public void addAll(
      final List<String> idsArg,
      final List<Embedding> embeddingsArg,
      final List<TextSegment> embeddedArg) {

    final int maxSize = maxSize(idsArg, embeddingsArg, embeddedArg);

    final List<String> ids = ensureSize(idsArg, maxSize);
    final List<Embedding> embeddings = ensureSize(embeddingsArg, maxSize);
    final List<TextSegment> embedded = ensureSize(embeddedArg, maxSize);

    for (int i = 0; i < maxSize; i++) {
      try {
        add(ids.get(i), embeddings.get(i), embedded.get(i));
      } catch (final Exception e) {
        log.error("Unable to add embeddings in Lucene", e);
      }
    }
  }

  /** {@inheritDoc} */
  @Override
  public EmbeddingSearchResult<TextSegment> search(final EmbeddingSearchRequest request) {
    throw new UnsupportedOperationException("Not supported yet.");
  }

  private <P> List<P> ensureSize(final List<P> provided, final int maxSize) {
    final List<P> sizedList;
    if (isNullOrEmpty(provided)) {
      sizedList = new ArrayList<>(Collections.nCopies(maxSize, null));
    } else {
      sizedList = new ArrayList<>(provided);
    }
    final int size = sizedList.size();
    if (size < maxSize) {
      sizedList.addAll(Collections.nCopies(maxSize - size, null));
    }
    return sizedList;
  }

  private boolean isBlank(final String text) {
    return text == null || text.isBlank();
  }

  private int maxSize(
      final List<String> idsArg,
      final List<Embedding> embeddingsArg,
      final List<TextSegment> embeddedArg) {
    int maxLen = 0;
    if (!isNullOrEmpty(idsArg)) {
      final int size = idsArg.size();
      if (maxLen < size) {
        maxLen = size;
      }
    }
    if (!isNullOrEmpty(embeddingsArg)) {
      final int size = embeddingsArg.size();
      if (maxLen < size) {
        maxLen = size;
      }
    }
    if (!isNullOrEmpty(embeddedArg)) {
      final int size = embeddedArg.size();
      if (maxLen < size) {
        maxLen = size;
      }
    }
    return maxLen;
  }

  private Field toField(final Entry<String, Object> entry) {
    final String fieldName = entry.getKey();
    final var fieldValue = entry.getValue();
    Field field;
    if (fieldValue instanceof final String string) {
      field = new StringField(fieldName, string, Store.YES);
    } else if (fieldValue instanceof final Integer number) {
      field = new IntField(fieldName, number, Store.YES);
    } else if (fieldValue instanceof final Long number) {
      field = new LongField(fieldName, number, Store.YES);
    } else if (fieldValue instanceof final Float number) {
      field = new FloatField(fieldName, number, Store.YES);
    } else if (fieldValue instanceof final Double number) {
      field = new DoubleField(fieldName, number, Store.YES);
    } else {
      field = new StringField(fieldName, String.valueOf(fieldValue), Store.YES);
    }
    return field;
  }
}