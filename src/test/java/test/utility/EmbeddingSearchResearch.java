package test.utility;

import dev.langchain4j.store.embedding.CosineSimilarity;
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
import org.junit.jupiter.api.MethodOrderer.OrderAnnotation;
import org.junit.jupiter.api.Order;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestMethodOrder;
import us.fatehi.search.DirectoryFactory;
import us.fatehi.search.LuceneEmbeddingStore;

@TestMethodOrder(OrderAnnotation.class)
public class EmbeddingSearchResearch {

    private static final TextEmbedding[] hits = {
        TextEmbedding.fromResource("hitDoc1.txt"),
        TextEmbedding.fromResource("hitDoc2.txt"),
        TextEmbedding.fromResource("hitDoc3.txt"),
    };
    private static final TextEmbedding[] misses = {
        TextEmbedding.fromResource("missDoc1.txt"),
    };
    private static final TextEmbedding[] queries = {
        TextEmbedding.fromResource("query1.txt"),
        TextEmbedding.fromResource("query2.txt"),
        TextEmbedding.fromResource("query3.txt"),
    };

    private Directory directory;
    private LuceneEmbeddingStore indexer;

    @Test
    @Order(1)
    public void _01_fullText() throws Exception {
        System.out.println("\n>> Full text query");

        for (final TextEmbedding stringEmbedding : hits) {
            indexer.add(stringEmbedding.id(), stringEmbedding.text());
        }
        for (final TextEmbedding stringEmbedding : misses) {
            indexer.add(stringEmbedding.text());
        }

        for (final TextEmbedding query : queries) {
            final org.apache.lucene.search.Query luceneQuery = buildQuery(query);
            retrieve(luceneQuery);
        }
    }

    @Test
    @Order(2)
    public void _02_cosine() throws Exception {
        System.out.println("\n>> Cosine similarities");

        for (final TextEmbedding query : queries) {
            System.out.printf("%n{\"query\"=\"%s\"}%n", query.text().text());
            final SortedMap<Double, String> map = new TreeMap<>((a, b) -> b.compareTo(a));
            for (final TextEmbedding stringEmbedding : hits) {
                final double similarity = CosineSimilarity.between(query.embedding(), stringEmbedding.embedding());
                map.put(similarity, stringEmbedding.text().text());
            }
            for (final TextEmbedding stringEmbedding : misses) {
                final double similarity = CosineSimilarity.between(query.embedding(), stringEmbedding.embedding());
                map.put(similarity, stringEmbedding.text().text());
            }
            for (final Entry<Double, String> entry : map.entrySet()) {
                System.out.printf("{\"score\"=\"%f\", \"text\"=\"%s\"}%n", entry.getKey(), entry.getValue());
            }
        }
    }

    @Test
    @Order(3)
    public void _03_embedding() throws Exception {
        System.out.println("\n>> Embedding vector query");

        for (final TextEmbedding stringEmbedding : hits) {
            indexer.add(stringEmbedding.id(), stringEmbedding.embedding());
        }
        for (final TextEmbedding stringEmbedding : misses) {
            indexer.add(stringEmbedding.id(), stringEmbedding.embedding());
        }

        for (final TextEmbedding query : queries) {
            final org.apache.lucene.search.Query luceneQuery = buildQuery(query);
            retrieve(luceneQuery);
        }
    }

    @Test
    @Order(4)
    public void _04_hybrid() throws Exception {
        System.out.println("\n>> Hybrid query");

        for (final TextEmbedding stringEmbedding : hits) {
            indexer.add(stringEmbedding.id(), stringEmbedding.embedding(), stringEmbedding.text());
        }
        for (final TextEmbedding stringEmbedding : misses) {
            indexer.add(stringEmbedding.id(), stringEmbedding.embedding(), stringEmbedding.text());
        }

        for (final TextEmbedding query : queries) {
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

    private org.apache.lucene.search.Query buildQuery(final TextEmbedding query) throws ParseException {
        System.out.printf("%n{\"query\"=\"%s\"}%n", query.text().text());
        org.apache.lucene.search.Query fullTextQuery;
        final QueryParser parser = new QueryParser("content", new StandardAnalyzer());
        fullTextQuery = parser.parse(query.text().text());

        final org.apache.lucene.search.Query vectorQuery =
                // new FloatVectorSimilarityQuery("embedding", query.embedding().vector(), 0.5f);
                new KnnFloatVectorQuery("embedding", query.embedding().vector(), 5);

        final BooleanQuery combinedQuery = new BooleanQuery.Builder()
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
                if (scoreDoc.score < 0.4) {
                    continue;
                }
                // Retrieve document contents
                final Document document = storedFields.document(scoreDoc.doc);
                final String id = document.get("id");
                final String content = TextEmbedding.fromResource(id).text().text();
                System.out.printf("{\"score\"=\"%f\", \"text\"=\"%s\"}%n", scoreDoc.score, content);
                hits.add(content);
            }
            return hits;
        }
    }
}
