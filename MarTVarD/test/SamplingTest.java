/**
 * Created by LoongKuan on 8/10/2016.
 */
public class SamplingTest {
    public static void main(String[] args) {
        int nAttributes = 5;
        int nValues = 5;
        double[] magnitudes = new double[]{0.5};
        int[] burnIns = new int[]{1000000};
        String[] type = new String[]{"prior", "posterior", "both"};

        String[] allFiles = new String[magnitudes.length * burnIns.length * type.length];

        for (int i = 0; i < magnitudes.length; i++) {
            for (int j = 0; j < burnIns.length; j++) {
                for (int k = 0; k < type.length; k++) {
                    int index = i * (burnIns.length * type.length) + j * type.length + k;
                    allFiles[index] = "n" + burnIns[j] + "_m" + magnitudes[i] + "_" + type[k];
                }
            }
        }

        String folder = "synthetic_" + nAttributes + "Att_" + nValues + "Val";

        //MainTest.standardAll(new int[]{1,6}, allFiles, -1, folder);
        MainTest.standardAll(new int[]{1,6}, allFiles, 1000, folder);
    }
}
