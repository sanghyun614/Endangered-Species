# 동물 멸종 위기 등급 예측

학부생 연구 프로그램으로 진행한 프로젝트입니다. [리포트](https://github.com/sanghyun614/Endangered-Species/blob/master/%EB%8F%99%EB%AC%BC%20%EB%A9%B8%EC%A2%85%20%EC%9C%84%EA%B8%B0%20%EC%98%88%EC%B8%A1%20%EB%A6%AC%ED%8F%AC%ED%8A%B8.pdf)

## Overview
수많은 동물의 멸종 위기 단계를 일일이 추적하고, 단계 변화 시기를 종별로 연구하여 예측하는 것은 시간이 오래 걸릴뿐더러 사람의 손길이 닿지 않는 공간에는 미처 파악하지 못하는 종들이 생겨날 수도 있습니다. 하나의 종을 연구하고 추적하는 데 걸리는 시간은 짧게는 한 달, 길게는 몇 개월이 걸린다고 합니다.

조사를 진행하는 도중에도 개체 수는 변화할 것이고, 멸종 위기 동물들에 대해서는 지속적이고 빠른 조사를 기반으로 등급이 매겨져야 합니다. 본 연구를 통해 현장 조사 혹은 관찰 이외에도 활용할 수 있는 데이터를 이용하여 멸종 위기종을 발견해 냄으로 동물 조사의 시간적/공간적 비용 감소를 기대했습니다.

## Contents
분석 과정은 아래와 같이 요약됩니다.

1. 데이터 수집
   * IUCN 웹 크롤링
2. 데이터 전처리
   * 변수 생성
   * 비대칭 데이터 처리
3. 예측 모형 적합 
