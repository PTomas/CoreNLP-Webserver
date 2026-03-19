package main;

import io.javalin.Javalin;

import edu.stanford.nlp.pipeline.*;
import edu.stanford.nlp.ling.*;
import edu.stanford.nlp.sentiment.SentimentCoreAnnotations;
import edu.stanford.nlp.neural.rnn.RNNCoreAnnotations;

import edu.stanford.nlp.trees.*;
import edu.stanford.nlp.util.CoreMap;

// import org.ejml.simple.SimpleMatrix;

import java.util.*;

public class CustomSentimentApi {

    public static void main(String[] args) {

        Properties props = new Properties();
        props.setProperty("annotators", "tokenize,ssplit,pos,parse,sentiment");

        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);

        int port = Integer.parseInt(System.getenv().getOrDefault("PORT", "7000"));
        Javalin app = Javalin.create().start(port);

        app.get("/sentiment", ctx -> {

            String text = ctx.queryParam("text");

            if (text == null || text.trim().isEmpty()) {
                ctx.status(400).result("Missing `text` parameter.");
                return;
            }

            try {

                Annotation doc = new Annotation(text);
                pipeline.annotate(doc);

                List<List<String>> allDependencies = new ArrayList<>();

                for (CoreMap sentence :
                        doc.get(CoreAnnotations.SentencesAnnotation.class)) {

                    List<CoreLabel> tokens =
                            sentence.get(CoreAnnotations.TokensAnnotation.class);

                    Tree sentimentTree =
                            sentence.get(SentimentCoreAnnotations.SentimentAnnotatedTree.class);

                    List<Tree> leaves = sentimentTree.getLeaves();

                    Map<Integer, String> tokenInfo = new HashMap<>();

                    for (int i = 0; i < tokens.size(); i++) {

                        CoreLabel token = tokens.get(i);

                        String word =
                                token.get(CoreAnnotations.TextAnnotation.class);

                        String pos =
                                token.get(CoreAnnotations.PartOfSpeechAnnotation.class);

                        int sentiment = 2;

                        for (Tree leaf : leaves) {

                            if (leaf.label().value().equals(word)) {

                                Tree parent = leaf.parent(sentimentTree);

                                if (parent != null) {
                                    sentiment =
                                            RNNCoreAnnotations.getPredictedClass(parent);
                                }

                                break;
                            }
                        }

                        tokenInfo.put(
                                i + 1,
                                String.format("%s/%s/%d", word, pos, sentiment)
                        );
                    }

                    Tree parseTree =
                            sentence.get(TreeCoreAnnotations.TreeAnnotation.class);

                    GrammaticalStructure gs =
                            new EnglishGrammaticalStructure(parseTree);

                    Collection<TypedDependency> dependencies =
                            gs.typedDependencies();

                    List<String> resultList = new ArrayList<>();

                    for (TypedDependency td : dependencies) {

                        String rel = td.reln().toString();

                        int govIndex = td.gov().index();
                        int depIndex = td.dep().index();

                        String gov =
                                tokenInfo.getOrDefault(govIndex, td.gov().word());

                        String dep =
                                tokenInfo.getOrDefault(depIndex, td.dep().word());

                        resultList.add(
                                String.format("%s(%s, %s)", rel, gov, dep)
                        );
                    }

                    allDependencies.add(resultList);
                }

                Map<String, Object> result = new HashMap<>();

                result.put("dependencies", allDependencies);

                ctx.json(result);

            }

            catch (Exception e) {

                e.printStackTrace();

                ctx.status(500).result("Internal Server Error: " + e.getMessage());
            }

        });

    }
}