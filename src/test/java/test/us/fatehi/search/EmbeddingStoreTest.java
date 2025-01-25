package test.us.fatehi.search;

import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2q.AllMiniLmL6V2QuantizedEmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.EmbeddingStoreIT;
import us.fatehi.search.LuceneEmbeddingStore;

public class EmbeddingStoreTest extends EmbeddingStoreIT {

  EmbeddingModel embeddingModel = new AllMiniLmL6V2QuantizedEmbeddingModel();
  EmbeddingStore<TextSegment> embeddingStore = LuceneEmbeddingStore.builder().build();

  @Override
  protected EmbeddingStore<TextSegment> embeddingStore() {
    return embeddingStore;
  }

  @Override
  protected EmbeddingModel embeddingModel() {
    return embeddingModel;
  }
}
