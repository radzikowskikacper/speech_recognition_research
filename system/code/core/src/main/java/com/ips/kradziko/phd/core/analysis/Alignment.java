package com.ips.kradziko.phd.core.analysis;

import org.javatuples.Quintet;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * Created by kradziko on 17/02/20.
 */



public class Alignment {
    private Map<String, Quintet<String, Integer, Double, Double, Double>> data;
    private Set<String> set1 = new HashSet<>();

    public void loadData(String data_dir) throws IOException {
        data = new HashMap<>();

        for(String filename : new String[]{"sentence1.tab", "word1.tab"}){//, "sentence3.tab"}){
            BufferedReader br = new BufferedReader(new FileReader(data_dir + "/doc/JEcontent/tab/" + filename));
            String sCurrentLine;

            while ((sCurrentLine = br.readLine()) != null) {
                String[] parts = sCurrentLine.split("\\.wav");
                parts[0] = parts[0].trim();
                parts[0] += ".wav";
                data.putIfAbsent(parts[0],
                        new Quintet<>(parts[parts.length - 1].trim(), -1, -1.0, -1.0, -1.0));
            }
        }

        Files.walk(Paths.get(data_dir + "/lbl/")).filter(Files::isRegularFile)
                .forEach(file -> {
                    if(!(file.toString().contains("sentence") || file.toString().contains("word"))
                            || !file.toString().contains("scores")) return;

                    String[] parts = file.toString().split("/");
                    int gender = parts[parts.length - 3].contains("female") ? 0 : 1;
                    int pos = 2;
                    if(parts[parts.length - 2].contains("segmental")) pos = 0;
                    else if(parts[parts.length - 2].contains("intonation") || parts[parts.length - 2].contains("accent")) pos = 1;

                    BufferedReader br = null;
                    String sCurrentLine;
                    try {
                        br = new BufferedReader(new FileReader(file.toString()));
                        while ((sCurrentLine = br.readLine()) != null) {
                            if(sCurrentLine.charAt(0) == '#') continue;

                            parts = sCurrentLine.split("\\s+");
                            double c = 0;
                            for(int i = parts.length - 1; i > 0; --i) c += Double.valueOf(parts[i]);
                            c /= parts.length - 1;

                            parts = parts[0].split("/");
                            set1.add(parts[2]);
                            data.put(parts[2], data.get(parts[2]).setAt1(gender));
                            switch(pos){
                                case 0: data.put(parts[2], data.get(parts[2]).setAt2(c)); break;
                                case 1: data.put(parts[2], data.get(parts[2]).setAt3(c)); break;
                                default: data.put(parts[2], data.get(parts[2]).setAt4(c));
                            }
                        }
                    } catch (FileNotFoundException e) {
                        e.printStackTrace();
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                });
    }
}
