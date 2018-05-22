package com.prmfincon.deeplearning;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.NumberedFileInputSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.eval.ROC;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.primitives.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Created by Pratyay on 5/22/2018.
 */
public class VibrationPredictionTest {
    private static final Logger log = LoggerFactory.getLogger(VibrationPredictionTest.class);

    static File baseDir = new File("src/main/resources/result/");
    static File trainDir = new File(baseDir,"train");
    static File featuresDirTrain = new File(trainDir,"features"); // path to directory containing feature files
    static File labelsDirTrain = new File(trainDir,"labels");
    static File testDir = new File(baseDir,"test");
    static File featuresDirTest = new File(testDir,"features"); // path to directory containing feature files
    static File labelsDirTest = new File(testDir,"labels");

    public static void main(String args[]){

        //Data and ETL

        final int NB_TRAIN_EXAMPLES = 172800*75/100;
        /*baseDir.mkdir();
        featuresDir.mkdir();
        labelsDir.mkdir();*/

        try {
            getCSVData();
            SequenceRecordReader trainFeatures = new CSVSequenceRecordReader(1, ",");  // number of rows to skip + delimiter
            trainFeatures.initialize(new NumberedFileInputSplit(featuresDirTrain.getAbsolutePath() + "/%d.csv", 0, NB_TRAIN_EXAMPLES - 1));
            SequenceRecordReader trainLabels = new CSVSequenceRecordReader(1, ",");
            trainLabels.initialize(new NumberedFileInputSplit(labelsDirTrain.getAbsolutePath() + "/%d.csv", 0, NB_TRAIN_EXAMPLES - 1));
            int numLabelClasses = 8;
            int BATCH_SIZE = 10;

            DataSetIterator trainData = new SequenceRecordReaderDataSetIterator(trainFeatures, trainLabels,
                    BATCH_SIZE, numLabelClasses, false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);
            //Normalize the training data
            DataNormalization normalizer = new NormalizerStandardize();
            normalizer.fit(trainData);              //Collect training data statistics
            trainData.reset();

            //Use previously collected statistics to normalize on-the-fly. Each DataSet returned by 'trainData' iterator will be normalized
            trainData.setPreProcessor(normalizer);

            // ----- Load the test data -----
            //Same process as for the training data.
            SequenceRecordReader testFeatures = new CSVSequenceRecordReader();
            testFeatures.initialize(new NumberedFileInputSplit(featuresDirTest.getAbsolutePath() + "/%d.csv", 0, 43199));
            SequenceRecordReader testLabels = new CSVSequenceRecordReader();
            testLabels.initialize(new NumberedFileInputSplit(labelsDirTest.getAbsolutePath() + "/%d.csv", 0, 43199));

            DataSetIterator testData = new SequenceRecordReaderDataSetIterator(testFeatures, testLabels, BATCH_SIZE, numLabelClasses,
                    false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

            testData.setPreProcessor(normalizer);   //Note that we are using the exact same normalization process as the training data




            //Building a LSTM Neural Network
            final double LEARNING_RATE = 0.05;
            final int lstmLayerSize = 300;
            final int NB_INPUTS = 86;

            ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .updater(new Sgd(LEARNING_RATE))
                    .graphBuilder()
                    .addInputs("trainFeatures")
                    .setOutputs("predictVibrations")
                    .addLayer("L1", new GravesLSTM.Builder()
                            .nIn(NB_INPUTS)
                            .nOut(lstmLayerSize)
                            .activation(Activation.SOFTSIGN)
                            .weightInit(WeightInit.DISTRIBUTION)
                            .build(), "trainFeatures")
                    .addLayer("predictVibrations", new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                            .activation(Activation.SOFTMAX)
                            .weightInit(WeightInit.DISTRIBUTION)
                            .nIn(lstmLayerSize).nOut(numLabelClasses).build(),"L1")
                    .pretrain(false).backprop(true)
                    .build();

            ComputationGraph model = new ComputationGraph(conf);
            model.init();


            //Training and Evaluating a LSTM Neural Network
            final int NB_EPOCHS = 100;

            for( int i=0; i<NB_EPOCHS; i++ ){
                model.fit(trainData);
            }

            ROC roc = new ROC();
            while (trainData.hasNext()) {
                DataSet batch = trainData.next();
                INDArray[] output = model.output(batch.getFeatures());
                roc.evalTimeSeries(batch.getLabels(), output[0]);
            }
            log.info("FINAL TEST AUC: " + roc.calculateAUC());
        } catch (Exception e){
            e.printStackTrace();
        }
    }

    private static void getCSVData() throws Exception {
        if (baseDir.exists()) return;    //Data already exists, don't download it again

        //RecordReader recordReader = new CSVRecordReader(1,",");
        //recordReader.initialize(new FileSplit(new ClassPathResource("aggregated_vibration.csv").getFile()));

        //Create directories
        baseDir.mkdir();
        trainDir.mkdir();
        featuresDirTrain.mkdir();
        labelsDirTrain.mkdir();
        testDir.mkdir();
        featuresDirTest.mkdir();
        labelsDirTest.mkdir();



        String data = IOUtils.toString(new ClassPathResource("aggregated_vibration.csv").getFile().toURI());

        String[] lines = data.split("\n");

        int lineCount = 0;
        List<Pair<String, Integer>> contentAndLabels = new ArrayList<>();
        for (String line : lines) {
            String transposed = line.replaceAll(" +", "\n");

            //Labels: first 100 examples (lines) are label 0, second 100 examples are label 1, and so on
            contentAndLabels.add(new Pair<>(transposed, lineCount++ / 100));
        }


        //Randomize and do a train/test split:
        Collections.shuffle(contentAndLabels, new Random(12345));

        int nTrain = 172800*75/100;   //75% train, 25% test
        int trainCount = 0;
        int testCount = 0;
        for (Pair<String, Integer> p : contentAndLabels) {
            //Write output in a format we can read, in the appropriate locations
            File outPathFeatures;
            File outPathLabels;
            if (trainCount < nTrain) {
                outPathFeatures = new File(featuresDirTrain, trainCount + ".csv");
                outPathLabels = new File(labelsDirTrain, trainCount + ".csv");
                trainCount++;
            } else {
                outPathFeatures = new File(featuresDirTest, testCount + ".csv");
                outPathLabels = new File(labelsDirTest, testCount + ".csv");
                testCount++;
            }

            FileUtils.writeStringToFile(outPathFeatures, p.getFirst());
            FileUtils.writeStringToFile(outPathLabels, p.getSecond().toString());
        }
    }
}
